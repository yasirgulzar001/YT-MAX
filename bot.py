#!/usr/bin/env python3
"""
Feature‑rich Telegram media bot.
Supports YouTube downloads, voice transcription (Whisper), TTS (Edge‑TTS),
inline queries, playlists, trimming, and daily limits.
"""

import asyncio
import io
import logging
import os
import pathlib
import re
import shutil
import signal
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import aiosqlite
import edge_tts
import yt_dlp
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQueryResultArticle,
    InputTextMessageContent,
    Update,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    InlineQueryHandler,
    MessageHandler,
    filters,
)

# ------------------------------------------------------------------------
# Configuration – all secrets and tunables live in environment variables
# ------------------------------------------------------------------------
BOT_TOKEN: str = os.environ["BOT_TOKEN"]
ADMIN_IDS: set[int] = set(
    int(uid) for uid in os.environ.get("ADMIN_IDS", "").split(",") if uid.strip()
)
DB_PATH: str = os.environ.get("DB_PATH", "bot_data.db")
WHISPER_MODEL: str = os.environ.get("WHISPER_MODEL", "base")
WHISPER_DEVICE: str = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE: str = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
DOWNLOAD_LIMIT_PER_DAY: int = int(os.environ.get("DOWNLOAD_LIMIT", "5"))
TRANSCRIBE_LIMIT_PER_DAY: int = int(os.environ.get("TRANSCRIBE_LIMIT", "10"))
TTS_LIMIT_PER_DAY: int = int(os.environ.get("TTS_LIMIT", "10"))
TRANSFER_SH_URL: str = "https://transfer.sh"
MAX_DIRECT_FILE_SIZE: int = 50 * 1024 * 1024  # 50 MiB
LOG_FILE: str = os.environ.get("LOG_FILE", "bot.log")

# ------------------------------------------------------------------------
# Logging – file + console, structured
# ------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------
# Shared state & lazy models
# ------------------------------------------------------------------------
user_languages: Dict[int, str] = {}           # user_id -> language code (e.g. "en")
whisper_model = None                          # loaded by _load_whisper_model()
whisper_lock = asyncio.Lock()                 # guards first‑load handshake

# ------------------------------------------------------------------------
# Database helpers (aiosqlite – all queries are async)
# ------------------------------------------------------------------------

@asynccontextmanager
async def db_connect():
    """Async context manager for aiosqlite connection with WAL mode."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL;")
        yield db


async def init_db() -> None:
    """Create tables if they don't exist."""
    async with db_connect() as db:
        await db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            joined_at TEXT DEFAULT (datetime('now')),
            banned INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS usage (
            user_id INTEGER,
            date TEXT,               -- YYYY-MM-DD
            downloads INTEGER DEFAULT 0,
            transcriptions INTEGER DEFAULT 0,
            tts INTEGER DEFAULT 0,
            PRIMARY KEY (user_id, date)
        );
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT,
            details TEXT,
            timestamp TEXT DEFAULT (datetime('now'))
        );
        """)
        await db.commit()


async def fetch_all(db: aiosqlite.Connection, query: str, params: Optional[tuple] = None) -> List[aiosqlite.Row]:
    """Shortcut for cursor.execute + fetchall."""
    cursor = await db.execute(query, params or ())
    return await cursor.fetchall()


async def ensure_user(db: aiosqlite.Connection, user: Any) -> bool:
    """Register user if new; returns False if banned."""
    rows = await fetch_all(db, "SELECT banned FROM users WHERE user_id = ?", (user.id,))
    if not rows:
        await db.execute(
            "INSERT INTO users (user_id, username, first_name) VALUES (?, ?, ?)",
            (user.id, user.username, user.first_name),
        )
        await db.commit()
        return True
    if rows[0]["banned"]:
        return False
    return True


async def get_usage(db: aiosqlite.Connection, user_id: int, today: str) -> Dict[str, int]:
    """Return {downloads, transcriptions, tts} for a user on a given date."""
    rows = await fetch_all(
        db,
        "SELECT downloads, transcriptions, tts FROM usage WHERE user_id = ? AND date = ?",
        (user_id, today),
    )
    if rows:
        row = rows[0]
        return {"downloads": row["downloads"], "transcriptions": row["transcriptions"], "tts": row["tts"]}
    return {"downloads": 0, "transcriptions": 0, "tts": 0}


async def increment_usage(db: aiosqlite.Connection, user_id: int, today: str, field: str) -> None:
    """Increment a usage counter by 1."""
    await db.execute(
        f"""
        INSERT INTO usage (user_id, date, downloads, transcriptions, tts)
        VALUES (?, ?, 0, 0, 0)
        ON CONFLICT(user_id, date) DO UPDATE SET {field} = {field} + 1
        """,
        (user_id, today),
    )
    await db.commit()


async def add_history(db: aiosqlite.Connection, user_id: int, action: str, details: str = "") -> None:
    """Insert an action into history and keep only the last 10 per user."""
    await db.execute(
        "INSERT INTO history (user_id, action, details) VALUES (?, ?, ?)",
        (user_id, action, details),
    )
    # keep only last 10
    await db.execute(
        """
        DELETE FROM history WHERE user_id = ? AND id NOT IN (
            SELECT id FROM history WHERE user_id = ? ORDER BY id DESC LIMIT 10
        )
        """,
        (user_id, user_id),
    )
    await db.commit()


# ------------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------------
def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for unit in ("KB", "MB", "GB"):
        n /= 1024.0
        if n < 1024.0:
            return f"{n:.1f} {unit}"
    return f"{n:.1f} TB"


def format_progress(d: dict) -> str:
    """Return a pretty progress string from a yt‑dlp progress dict."""
    pct = d.get("percent", 0)
    eta = d.get("eta", "?")
    speed = d.get("speed", "?")
    downloaded = d.get("downloaded_bytes", 0)
    total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
    bar_length = 15
    filled = int(round(pct * bar_length / 100)) if pct else 0
    bar = "█" * filled + "░" * (bar_length - filled)
    size_str = f"{human_size(downloaded)}"
    if total:
        size_str += f"/{human_size(total)}"
    return f"⬇️ {pct:.1f}% |{bar}| {size_str} @ {speed}\n⏳ ETA: {eta}"


def parse_time(s: str) -> Optional[float]:
    """Parse mm:ss or seconds string to float seconds."""
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 2:
            try:
                mins, secs = int(parts[0]), float(parts[1])
                return mins * 60 + secs
            except ValueError:
                return None
        return None
    try:
        return float(s)
    except ValueError:
        return None


async def safe_edit(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, **kwargs) -> None:
    """Edit the message if from a callback, otherwise reply."""
    if update.callback_query:
        await update.callback_query.edit_message_text(text, **kwargs)
    else:
        if update.message:
            await update.message.reply_text(text, **kwargs)


async def temporary_message(context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str, delay: int = 10) -> None:
    """Send a message that deletes itself after `delay` seconds."""
    msg = await context.bot.send_message(chat_id, text)
    await asyncio.sleep(delay)
    with suppress(Exception):
        await msg.delete()


# ------------------------------------------------------------------------
# Transfer.sh upload
# ------------------------------------------------------------------------
async def upload_transfer(session: aiohttp.ClientSession, file_path: pathlib.Path) -> Optional[str]:
    """Upload a file to transfer.sh and return the download link."""
    try:
        data = file_path.read_bytes()
        for attempt in range(2):
            try:
                async with session.put(
                    f"{TRANSFER_SH_URL}/{file_path.name}",
                    data=data,
                    headers={"Max-Downloads": "1", "Max-Days": "1"},
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status == 200:
                        return (await resp.text()).strip()
                    logger.warning("transfer.sh upload attempt %d returned %d", attempt, resp.status)
            except Exception:
                if attempt == 1:
                    raise
                await asyncio.sleep(1)
    except Exception as e:
        logger.error("upload_transfer failed: %s", e)
    return None


# ------------------------------------------------------------------------
# Whisper model loading (lazy, thread‑safe)
# ------------------------------------------------------------------------
def _load_whisper_model():
    """Synchronous model loader – runs in executor."""
    from faster_whisper import WhisperModel
    model = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )
    return model


async def get_whisper_model():
    """Return the whisper model, loading it exactly once."""
    global whisper_model
    if whisper_model is not None:
        return whisper_model
    async with whisper_lock:
        if whisper_model is not None:  # double‑check inside lock
            return whisper_model
        loop = asyncio.get_running_loop()
        whisper_model = await loop.run_in_executor(None, _load_whisper_model)
        logger.info("Whisper model loaded: %s", WHISPER_MODEL)
        return whisper_model


async def preload_whisper():
    """Background task to warm the model."""
    logger.info("Pre‑loading Whisper model in background…")
    await get_whisper_model()
    logger.info("Whisper model ready.")


def transcribe_audio_sync(file_data: bytes, language: Optional[str] = None) -> str:
    """
    Transcribe audio bytes (opus/wav). Must be called from a thread
    because faster‑whisper's transcribe is blocking.
    """
    # Access the already‑loaded global model – safe because we ensure it's loaded
    # before dispatching to a thread.
    model = whisper_model
    if model is None:
        raise RuntimeError("Whisper model not loaded yet")

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp.write(file_data)
        tmp_path = tmp.name
    try:
        segments, _ = model.transcribe(tmp_path, language=language, beam_size=5)
        text = " ".join(seg.text for seg in segments)
        return text or "(silence / no speech detected)"
    finally:
        with suppress(OSError):
            os.unlink(tmp_path)


# ------------------------------------------------------------------------
# TTS via edge‑tts
# ------------------------------------------------------------------------
VOICE_OPTIONS = [
    ("en-US-AriaNeural", "🇺🇸 English US (Female)"),
    ("en-US-GuyNeural", "🇺🇸 English US (Male)"),
    ("en-GB-SoniaNeural", "🇬🇧 English UK (Female)"),
    ("en-GB-RyanNeural", "🇬🇧 English UK (Male)"),
    ("es-ES-AlvaroNeural", "🇪🇸 Spanish (Male)"),
    ("es-MX-DaliaNeural", "🇲🇽 Spanish MX (Female)"),
    ("fr-FR-DeniseNeural", "🇫🇷 French (Female)"),
    ("fr-CA-JeanNeural", "🇨🇦 French CA (Male)"),
    ("de-DE-KatjaNeural", "🇩🇪 German (Female)"),
    ("de-DE-ConradNeural", "🇩🇪 German (Male)"),
    ("it-IT-ElsaNeural", "🇮🇹 Italian"),
    ("pt-BR-FranciscaNeural", "🇧🇷 Portuguese BR"),
    ("ru-RU-SvetlanaNeural", "🇷🇺 Russian"),
    ("ja-JP-NanamiNeural", "🇯🇵 Japanese"),
    ("ko-KR-SunHiNeural", "🇰🇷 Korean"),
]


async def generate_tts(text: str, voice: str) -> bytes:
    """Generate speech MP3 bytes using edge‑tts."""
    communicate = edge_tts.Communicate(text, voice)
    mp3_data = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_data.write(chunk["data"])
    return mp3_data.getvalue()


# ------------------------------------------------------------------------
# YouTube / yt‑dlp helpers
# ------------------------------------------------------------------------
def _search_youtube(query: str, limit: int = 5) -> List[dict]:
    """Synchronous YouTube search."""
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "force_generic_extractor": False,
        "default_search": "ytsearch",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
        return info.get("entries", []) or []


def _get_playlist_info(url: str) -> List[dict]:
    """Extract playlist entries (flat)."""
    ydl_opts = {"quiet": True, "no_warnings": True, "extract_flat": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info.get("entries", []) or []


# ------------------------------------------------------------------------
# Core download & progress logic
# ------------------------------------------------------------------------
async def progress_updater(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    message_id: int,
    queue: asyncio.Queue,
) -> None:
    """Edit the progress message until a None sentinel is received."""
    while True:
        try:
            progress_dict = await asyncio.wait_for(queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        if progress_dict is None:
            break
        text = format_progress(progress_dict)
        try:
            await context.bot.edit_message_text(
                text, chat_id=chat_id, message_id=message_id
            )
        except Exception as e:
            if "message is not modified" not in str(e):
                logger.debug("Progress edit failed: %s", e)


def _yt_download_progress_hook(queue: asyncio.Queue):
    """Return a closure that pushes progress dicts into the queue."""
    def hook(d):
        asyncio.run_coroutine_threadsafe(queue.put(d), asyncio.get_event_loop())
    return hook


async def download_and_send(
    context: ContextTypes.DEFAULT_TYPE,
    update_or_query,
    url: str,
    format_spec: str = "bestvideo+bestaudio/best",
    trim_start: Optional[float] = None,
    trim_end: Optional[float] = None,
    is_audio: bool = False,
) -> bool:
    """
    Download media, show progress, send file or upload link.
    Returns True on success.
    """
    # Determine chat_id and progress message
    if update_or_query.callback_query:
        progress_msg = update_or_query.callback_query.message
        chat_id = progress_msg.chat_id
    else:
        progress_msg = await update_or_query.message.reply_text("⏳ Preparing download…")
        chat_id = update_or_query.message.chat_id
    message_id = progress_msg.message_id

    temp_dir = tempfile.mkdtemp(prefix="ytdl_")
    queue: asyncio.Queue = asyncio.Queue()
    updater_task = asyncio.create_task(
        progress_updater(context, chat_id, message_id, queue)
    )

    # Common yt‑dlp options
    base_opts = {
        "outtmpl": str(pathlib.Path(temp_dir) / "%(title).50s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4" if not is_audio else None,
        "progress_hooks": [_yt_download_progress_hook(queue)],
        # Use cookies if file exists (for age‑restricted / private videos)
        "cookiefile": "cookies.txt" if os.path.exists("cookies.txt") else None,
    }

    try:
        # Step 1: extract metadata (no download)
        info_opts = {**base_opts, "format": format_spec}
        with yt_dlp.YoutubeDL(info_opts) as ydl:
            loop = asyncio.get_running_loop()
            info = await loop.run_in_executor(None, ydl.extract_info, url, False)
            if not info:
                raise ValueError("Could not extract media info.")
            title = info.get("title", "media")
            duration = info.get("duration", 0)

        # Step 2: actual download (separate instance to avoid parameter hacking)
        download_opts = {**base_opts, "format": format_spec}
        with yt_dlp.YoutubeDL(download_opts) as ydl:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, ydl.download, [url])

        # Signal progress updater that download finished
        await queue.put(None)
        await updater_task

        # Find the downloaded file (newest in temp_dir)
        downloaded_files = sorted(
            pathlib.Path(temp_dir).iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if not downloaded_files:
            raise FileNotFoundError("No output file found.")
        file_path = downloaded_files[0]

        # Optional trimming
        if trim_start is not None or trim_end is not None:
            trimmed_path = pathlib.Path(temp_dir) / f"trimmed_{file_path.name}"
            cmd = ["ffmpeg", "-y", "-i", str(file_path)]
            if trim_start is not None:
                cmd.extend(["-ss", str(trim_start)])
            if trim_end is not None:
                cmd.extend(["-to", str(trim_end)])
            cmd.extend(["-c", "copy", str(trimmed_path)])
            proc = await asyncio.create_subprocess_exec(*cmd)
            await proc.communicate()
            if proc.returncode == 0 and trimmed_path.exists():
                file_path.unlink()
                file_path = trimmed_path
            else:
                logger.warning("Trimming failed – sending original file.")

        file_size = file_path.stat().st_size

        # Deliver file
        if file_size <= MAX_DIRECT_FILE_SIZE:
            with open(file_path, "rb") as f:
                if is_audio:
                    await context.bot.send_audio(chat_id, f, title=title)
                else:
                    await context.bot.send_video(chat_id, f, supports_streaming=True, caption=title)
            await context.bot.edit_message_text(
                f"✅ Sent: {title}", chat_id=chat_id, message_id=message_id
            )
        else:
            await context.bot.edit_message_text(
                "⬆️ Uploading to transfer.sh (large file)…", chat_id=chat_id, message_id=message_id
            )
            async with aiohttp.ClientSession() as session:
                link = await upload_transfer(session, file_path)
            if link:
                await context.bot.edit_message_text(
                    f"📎 [Download {title}]({link})",
                    chat_id=chat_id,
                    message_id=message_id,
                    parse_mode=ParseMode.MARKDOWN,
                )
            else:
                await context.bot.edit_message_text(
                    "❌ Upload failed and file too large to send directly.",
                    chat_id=chat_id,
                    message_id=message_id,
                )

        # Log history
        async with db_connect() as db:
            await add_history(db, chat_id, "download", f"{title} ({url})")
        return True

    except Exception as e:
        logger.exception("Download failed")
        await safe_edit(update_or_query, context, f"❌ Download failed: {e}")
        # Make sure updater stops
        await queue.put(None)
        await updater_task
        return False
    finally:
        # Cleanup temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        # Ensure updater is stopped even in edge cases
        if not updater_task.done():
            await queue.put(None)
            await updater_task


# ------------------------------------------------------------------------
# Handlers – commands, messages, callbacks
# ------------------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message."""
    user = update.effective_user
    async with db_connect() as db:
        if not await ensure_user(db, user):
            await update.message.reply_text("🚫 You are banned.")
            return
    await update.message.reply_text(
        "👋 Hello! I'm a versatile media bot.\n\n"
        "• Send a YouTube/URL – I'll offer to download video or audio.\n"
        "• Send a voice note or audio file – I'll transcribe it.\n"
        "• /tts <text> or send any text – I'll speak it (choose a voice).\n"
        "• /search <query> – search YouTube.\n"
        "• /playlist <url> – list playlist videos.\n"
        "• /language <code> – set transcription language (or auto).\n"
        "• /history – your last 10 actions.\n"
        "• /stats – your usage today.\n\n"
        "Inline mode: @bot <query/link>.\n"
        "Use /help for more."
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show help."""
    await update.message.reply_text(
        "🎛 *Commands & Tips*\n\n"
        "/start – Welcome message\n"
        "/help – This help\n"
        "/language `en`|`auto` – set transcription language\n"
        "/tts `text` – text‑to‑speech (default voice will be asked)\n"
        "/search `query` – YouTube search\n"
        "/playlist `url` – List videos in a playlist; you can download all or individual.\n"
        "/history – last 10 actions\n"
        "/stats – your daily usage\n\n"
        "Send a link → quality picker. For videos you can trim after download.\n"
        "Voice notes → transcribed automatically.\n"
        "Inline mode: type `@bot <query>` in any chat.",
        parse_mode=ParseMode.MARKDOWN,
    )


# ------------------------------------------------------------------------
# URL detection – quality picker
# ------------------------------------------------------------------------
URL_PATTERN = re.compile(r"https?://\S+")

async def handle_possible_url(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[bool]:
    """If the message contains a URL, offer download format choices."""
    text = update.message.text
    if not text:
        return None
    url_match = URL_PATTERN.search(text)
    if not url_match:
        return None   # not a URL – let TTS handler take over

    url = url_match.group()
    user_id = update.effective_user.id

    async with db_connect() as db:
        if not await ensure_user(db, update.effective_user):
            await update.message.reply_text("🚫 Banned.")
            return True
        usage = await get_usage(db, user_id, today_str())
        if usage["downloads"] >= DOWNLOAD_LIMIT_PER_DAY:
            await update.message.reply_text("⚠️ Daily download limit reached.")
            return True

    context.user_data["pending_url"] = url
    keyboard = [
        [InlineKeyboardButton("🎥 Best Video", callback_data="dl:bestvideo")],
        [InlineKeyboardButton("🎞 1080p", callback_data="dl:1080")],
        [InlineKeyboardButton("🎞 720p", callback_data="dl:720")],
        [InlineKeyboardButton("🎵 Audio only", callback_data="dl:audio")],
        [InlineKeyboardButton("✂️ Trim video", callback_data="dl:trim")],
    ]
    await update.message.reply_text(
        f"🔗 Found URL. Choose format:\n{url}",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )
    # Stop further handlers (text_message) from processing this update
    return True


async def callback_download_format(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle quality selection callbacks (but *not* dl:trim)."""
    query = update.callback_query
    await query.answer()
    data = query.data
    url = context.user_data.get("pending_url")
    if not url:
        await query.edit_message_text("❌ Session expired. Send the link again.")
        return

    format_map = {
        "dl:bestvideo": "bestvideo+bestaudio/best",
        "dl:1080": "bestvideo[height<=1080]+bestaudio/best",
        "dl:720": "bestvideo[height<=720]+bestaudio/best",
        "dl:audio": "bestaudio/best",
    }
    fmt = format_map[data]
    is_audio = (data == "dl:audio")
    await query.edit_message_text("⬇️ Downloading…")
    async with db_connect() as db:
        await increment_usage(db, update.effective_user.id, today_str(), "downloads")
    await download_and_send(context, update, url, fmt, is_audio=is_audio)
    context.user_data.pop("pending_url", None)


# ------------------------------------------------------------------------
# Trimming conversation (separate, triggered by dl:trim)
# ------------------------------------------------------------------------
TRIM_START, TRIM_END = range(2)

async def trim_start_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Callback when user clicks 'Trim video'."""
    query = update.callback_query
    url = context.user_data.get("pending_url")
    if not url:
        await query.edit_message_text("❌ Session expired.")
        return ConversationHandler.END
    await query.answer()
    await query.edit_message_text(
        "✂️ Enter start time (mm:ss or seconds) or `skip` for beginning:"
    )
    return TRIM_START


async def trim_start_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receive start time."""
    text = update.message.text.strip().lower()
    if text in ("skip", "0", "0:00"):
        context.user_data["trim_start"] = None
    else:
        start = parse_time(text)
        if start is None:
            await update.message.reply_text("⏱ Invalid format. Try `1:30` or `90`.")
            return TRIM_START
        context.user_data["trim_start"] = start
    await update.message.reply_text("⏱ Enter end time (mm:ss/seconds) or `skip` for end:")
    return TRIM_END


async def trim_end_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receive end time and start download+trim."""
    text = update.message.text.strip().lower()
    if text in ("skip",):
        context.user_data["trim_end"] = None
    else:
        end = parse_time(text)
        if end is None:
            await update.message.reply_text("⏱ Invalid format.")
            return TRIM_END
        context.user_data["trim_end"] = end

    url = context.user_data.get("pending_url")
    if not url:
        await update.message.reply_text("❌ Session expired.")
        return ConversationHandler.END
    start = context.user_data.get("trim_start")
    end = context.user_data.get("trim_end")
    await update.message.reply_text("⬇️ Downloading and trimming…")
    async with db_connect() as db:
        await increment_usage(db, update.effective_user.id, today_str(), "downloads")
    await download_and_send(
        context, update, url,
        format_spec="bestvideo+bestaudio/best",
        trim_start=start,
        trim_end=end,
    )
    # Clean up user data
    for key in ("pending_url", "trim_start", "trim_end"):
        context.user_data.pop(key, None)
    return ConversationHandler.END


# ------------------------------------------------------------------------
# Voice / audio transcription
# ------------------------------------------------------------------------
async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Transcribe voice messages or audio files."""
    user = update.effective_user
    async with db_connect() as db:
        if not await ensure_user(db, user):
            await update.message.reply_text("🚫 Banned.")
            return
        usage = await get_usage(db, user.id, today_str())
        if usage["transcriptions"] >= TRANSCRIBE_LIMIT_PER_DAY:
            await update.message.reply_text("⚠️ Daily transcription limit reached.")
            return

    # Download the audio file
    if update.message.voice:
        file = await update.message.voice.get_file()
    elif update.message.audio:
        file = await update.message.audio.get_file()
    else:
        return

    status_msg = await update.message.reply_text("🎤 Transcribing… (model may load)")
    try:
        file_bytes = await file.download_as_bytearray()
        # Ensure whisper model is loaded before passing to thread
        await get_whisper_model()
        language = user_languages.get(user.id)  # None = auto
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(
            None, transcribe_audio_sync, file_bytes, language
        )
        # Split if longer than 4000 chars (Telegram message limit)
        parts = []
        remaining = text
        while len(remaining) > 4000:
            split_pos = remaining.rfind(" ", 0, 3800)
            if split_pos == -1:
                split_pos = 3800
            parts.append(remaining[:split_pos])
            remaining = remaining[split_pos:].strip()
        parts.append(remaining)
        for part in parts:
            await update.message.reply_text(part)
        await status_msg.delete()
        # Update usage & history
        async with db_connect() as db:
            await increment_usage(db, user.id, today_str(), "transcriptions")
            await add_history(db, user.id, "transcription", text[:100])
    except Exception as e:
        logger.exception("Transcription error")
        await status_msg.edit_text(f"❌ Transcription failed: {e}")


# ------------------------------------------------------------------------
# Text‑to‑speech
# ------------------------------------------------------------------------
async def tts_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Direct /tts <text>."""
    text = " ".join(context.args)
    if not text:
        await update.message.reply_text("Usage: /tts <text>")
        return
    context.user_data["tts_text"] = text
    kb = _tts_voice_keyboard()
    await update.message.reply_text("🔊 Choose a voice:", reply_markup=kb)


async def text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Any non‑command, non‑URL text prompts TTS."""
    msg = update.message.text
    # This handler runs only if handle_possible_url did *not* claim the update
    context.user_data["tts_text"] = msg
    kb = _tts_voice_keyboard()
    await update.message.reply_text("🔊 Choose a voice:", reply_markup=kb)


def _tts_voice_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(name, callback_data=f"tts:{short}")]
        for short, name in VOICE_OPTIONS
    ]
    return InlineKeyboardMarkup(buttons)


async def tts_voice_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate and send TTS audio."""
    query = update.callback_query
    await query.answer()
    voice = query.data.split(":", 1)[1]
    text = context.user_data.get("tts_text")
    if not text:
        await query.edit_message_text("❌ No text. Send me something to speak.")
        return

    user_id = update.effective_user.id
    async with db_connect() as db:
        if not await ensure_user(db, update.effective_user):
            await query.edit_message_text("🚫 Banned.")
            return
        usage = await get_usage(db, user_id, today_str())
        if usage["tts"] >= TTS_LIMIT_PER_DAY:
            await query.edit_message_text("⚠️ TTS limit reached.")
            return

    await query.edit_message_text("🔊 Generating speech…")
    try:
        mp3_bytes = await generate_tts(text, voice)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(mp3_bytes)
            fname = f.name
        with open(fname, "rb") as f:
            await context.bot.send_audio(update.effective_chat.id, f, title="tts.mp3")
        os.unlink(fname)
        await query.edit_message_text("✅ Voice message sent.")
        async with db_connect() as db:
            await increment_usage(db, user_id, today_str(), "tts")
            await add_history(db, user_id, "tts", text[:100])
    except Exception as e:
        await query.edit_message_text(f"❌ TTS failed: {e}")


# ------------------------------------------------------------------------
# Language setting
# ------------------------------------------------------------------------
async def set_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set or view transcription language."""
    user_id = update.effective_user.id
    if not context.args:
        current = user_languages.get(user_id, "auto")
        await update.message.reply_text(
            f"Current language: {current}\nSend `/language en` to set or `/language auto`."
        )
        return
    lang = context.args[0].lower()
    if lang == "auto":
        user_languages.pop(user_id, None)
    else:
        user_languages[user_id] = lang
    await update.message.reply_text(f"🌐 Transcription language set to: {lang}")


# ------------------------------------------------------------------------
# Search & playlist
# ------------------------------------------------------------------------
async def search_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """YouTube search."""
    query = " ".join(context.args)
    if not query:
        await update.message.reply_text("Usage: /search <query>")
        return
    await update.message.reply_text("🔍 Searching…")
    loop = asyncio.get_running_loop()
    try:
        results = await loop.run_in_executor(None, _search_youtube, query, 5)
    except Exception as e:
        await update.message.reply_text(f"❌ Search error: {e}")
        return
    if not results:
        await update.message.reply_text("No results.")
        return
    buttons = []
    for i, entry in enumerate(results):
        title = entry.get("title", "No title")
        vid_id = entry.get("id")
        url = f"https://youtu.be/{vid_id}" if vid_id else entry.get("url", "")
        buttons.append([InlineKeyboardButton(f"{i+1}. {title[:50]}", callback_data=f"dl:search:{vid_id}")])
    await update.message.reply_text("Select to download:", reply_markup=InlineKeyboardMarkup(buttons))


async def search_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle selection from search results."""
    query = update.callback_query
    await query.answer()
    data = query.data
    if not data.startswith("dl:search:"):
        return
    vid_id = data.split(":", 2)[2]
    url = f"https://youtu.be/{vid_id}"
    context.user_data["pending_url"] = url
    keyboard = [
        [InlineKeyboardButton("🎥 Best Video", callback_data="dl:bestvideo")],
        [InlineKeyboardButton("🎵 Audio only", callback_data="dl:audio")],
    ]
    await query.edit_message_text(f"🔗 {url}\nChoose format:", reply_markup=InlineKeyboardMarkup(keyboard))


async def playlist_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List playlist entries and offer individual/all downloads."""
    url = " ".join(context.args)
    if not url or not URL_PATTERN.match(url):
        await update.message.reply_text("Usage: /playlist <url>")
        return
    await update.message.reply_text("📑 Fetching playlist…")
    loop = asyncio.get_running_loop()
    try:
        entries = await loop.run_in_executor(None, _get_playlist_info, url)
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")
        return
    if not entries:
        await update.message.reply_text("No videos found.")
        return
    context.user_data["playlist_entries"] = entries
    buttons = []
    for i, entry in enumerate(entries[:10]):  # show first 10 with buttons
        title = entry.get("title", "Unknown")
        vid_id = entry.get("id")
        if vid_id:
            buttons.append([InlineKeyboardButton(f"🎬 {title[:50]}", callback_data=f"pl:down:{vid_id}")])
    buttons.append([InlineKeyboardButton("📥 Download All (background)", callback_data="pl:all")])
    await update.message.reply_text("🎵 Playlist videos:", reply_markup=InlineKeyboardMarkup(buttons))


async def playlist_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle playlist action buttons."""
    query = update.callback_query
    await query.answer()
    data = query.data
    if data == "pl:all":
        entries = context.user_data.get("playlist_entries")
        if not entries:
            await query.edit_message_text("❌ Session expired.")
            return
        await query.edit_message_text("📥 Downloading entire playlist in background…")
        asyncio.create_task(_download_all_playlist(context, update.effective_chat.id, entries))
        return
    if data.startswith("pl:down:"):
        vid_id = data.split(":", 2)[2]
        url = f"https://youtu.be/{vid_id}"
        context.user_data["pending_url"] = url
        await query.edit_message_text(f"⬇️ Downloading {url}…")
        async with db_connect() as db:
            await increment_usage(db, update.effective_user.id, today_str(), "downloads")
        await download_and_send(context, update, url)
        return


async def _download_all_playlist(context: ContextTypes.DEFAULT_TYPE, chat_id: int, entries: List[dict]) -> None:
    """Background task: download every video in a playlist one by one."""
    total = len(entries)
    status_msg = await context.bot.send_message(chat_id, f"⏳ Starting playlist download (0/{total})…")
    done = 0
    for entry in entries:
        vid_id = entry.get("id")
        if not vid_id:
            continue
        url = f"https://youtu.be/{vid_id}"
        try:
            # Use a simplified download without progress UI
            temp_dir = tempfile.mkdtemp(prefix="pl_")
            ydl_opts = {
                "format": "bestvideo+bestaudio/best",
                "outtmpl": str(pathlib.Path(temp_dir) / "%(title).50s.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
                "merge_output_format": "mp4",
                "cookiefile": "cookies.txt" if os.path.exists("cookies.txt") else None,
            }
            loop = asyncio.get_running_loop()
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                await loop.run_in_executor(None, ydl.download, [url])
            # Send the first downloaded file
            files = sorted(pathlib.Path(temp_dir).iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if files:
                file_path = files[0]
                with open(file_path, "rb") as f:
                    await context.bot.send_video(chat_id, f, supports_streaming=True)
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            logger.warning("Playlist download failed for %s: %s", url, e)
        done += 1
        try:
            await status_msg.edit_text(f"⏳ Downloading playlist ({done}/{total})…")
        except Exception:
            pass
        await asyncio.sleep(0.5)  # avoid flooding
    await status_msg.edit_text(f"✅ Playlist completed ({total} videos).")


# ------------------------------------------------------------------------
# Inline mode
# ------------------------------------------------------------------------
async def inline_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline queries: direct link or search."""
    query_text = update.inline_query.query.strip()
    if not query_text:
        return
    if URL_PATTERN.match(query_text):
        url = query_text
        results = [
            InlineQueryResultArticle(
                id=uuid.uuid4().hex,
                title="Download this video",
                description=url,
                input_message_content=InputTextMessageContent(f"⬇️ Downloading: {url}"),
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Choose format", callback_data=f"inline_dl:{url}")]]
                ),
            )
        ]
        await update.inline_query.answer(results, cache_time=60)
        return
    # Search YouTube
    loop = asyncio.get_running_loop()
    try:
        results_list = await loop.run_in_executor(None, _search_youtube, query_text, 10)
    except Exception:
        return
    articles = []
    for entry in results_list:
        title = entry.get("title", "No title")
        vid_id = entry.get("id")
        if not vid_id:
            continue
        url = f"https://youtu.be/{vid_id}"
        articles.append(
            InlineQueryResultArticle(
                id=vid_id,
                title=title,
                description=url,
                input_message_content=InputTextMessageContent(f"🎬 {title}\n{url}"),
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Download", callback_data=f"inline_dl:{url}")]]
                ),
            )
        )
    await update.inline_query.answer(articles, cache_time=30)


async def inline_download_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Callback from inline result – show quality picker."""
    query = update.callback_query
    await query.answer()
    data = query.data
    if data.startswith("inline_dl:"):
        url = data.split(":", 1)[1]
        context.user_data["pending_url"] = url
        kb = [
            [InlineKeyboardButton("🎥 Best Video", callback_data="dl:bestvideo")],
            [InlineKeyboardButton("🎵 Audio only", callback_data="dl:audio")],
        ]
        await query.edit_message_text(f"🔗 {url}\nChoose format:", reply_markup=InlineKeyboardMarkup(kb))


# ------------------------------------------------------------------------
# History & stats
# ------------------------------------------------------------------------
async def history_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show last 10 actions."""
    user_id = update.effective_user.id
    async with db_connect() as db:
        rows = await fetch_all(
            db,
            "SELECT action, details, timestamp FROM history WHERE user_id = ? ORDER BY id DESC LIMIT 10",
            (user_id,),
        )
    if not rows:
        await update.message.reply_text("No history yet.")
        return
    lines = [f"• {r['timestamp']} | {r['action']}: {r['details'][:100]}" for r in rows]
    await update.message.reply_text("📋 Last actions:\n" + "\n".join(lines))


async def user_stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show daily usage."""
    user_id = update.effective_user.id
    today = today_str()
    async with db_connect() as db:
        usage = await get_usage(db, user_id, today)
    msg = (
        f"📊 Usage today ({today}):\n"
        f"Downloads: {usage['downloads']}/{DOWNLOAD_LIMIT_PER_DAY}\n"
        f"Transcriptions: {usage['transcriptions']}/{TRANSCRIBE_LIMIT_PER_DAY}\n"
        f"TTS: {usage['tts']}/{TTS_LIMIT_PER_DAY}"
    )
    await update.message.reply_text(msg)


# ------------------------------------------------------------------------
# Admin commands
# ------------------------------------------------------------------------
async def admin_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show global stats (admins only)."""
    if update.effective_user.id not in ADMIN_IDS:
        return
    today = today_str()
    async with db_connect() as db:
        total_users = (await fetch_all(db, "SELECT COUNT(*) as cnt FROM users"))[0]["cnt"]
        active_today = (
            await fetch_all(db, "SELECT COUNT(DISTINCT user_id) as cnt FROM usage WHERE date = ?", (today,))
        )[0]["cnt"]
        banned = (await fetch_all(db, "SELECT COUNT(*) as cnt FROM users WHERE banned=1"))[0]["cnt"]
    await update.message.reply_text(
        f"👥 Users: {total_users}\n📅 Active today: {active_today}\n🚫 Banned: {banned}"
    )


async def admin_broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message to all non‑banned users."""
    if update.effective_user.id not in ADMIN_IDS:
        return
    text = " ".join(context.args)
    if not text:
        await update.message.reply_text("Usage: /broadcast <message>")
        return
    async with db_connect() as db:
        rows = await fetch_all(db, "SELECT user_id FROM users WHERE banned=0")
    count = 0
    for row in rows:
        try:
            await context.bot.send_message(row["user_id"], text)
            count += 1
        except Exception:
            pass
    await update.message.reply_text(f"📣 Broadcast sent to {count} users.")


async def admin_ban(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Ban a user by ID."""
    if update.effective_user.id not in ADMIN_IDS:
        return
    if not context.args:
        await update.message.reply_text("Usage: /ban <user_id>")
        return
    uid = int(context.args[0])
    async with db_connect() as db:
        await db.execute("UPDATE users SET banned=1 WHERE user_id=?", (uid,))
        await db.commit()
    await update.message.reply_text(f"🚫 User {uid} banned.")


async def admin_unban(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Unban a user by ID."""
    if update.effective_user.id not in ADMIN_IDS:
        return
    if not context.args:
        await update.message.reply_text("Usage: /unban <user_id>")
        return
    uid = int(context.args[0])
    async with db_connect() as db:
        await db.execute("UPDATE users SET banned=0 WHERE user_id=?", (uid,))
        await db.commit()
    await update.message.reply_text(f"✅ User {uid} unbanned.")


# ------------------------------------------------------------------------
# Application setup & graceful shutdown
# ------------------------------------------------------------------------
def _setup_handlers(application: Application) -> None:
    """Register all handlers in the correct order."""
    # Commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("language", set_language))
    application.add_handler(CommandHandler("tts", tts_command))
    application.add_handler(CommandHandler("search", search_cmd))
    application.add_handler(CommandHandler("playlist", playlist_cmd))
    application.add_handler(CommandHandler("history", history_cmd))
    application.add_handler(CommandHandler("stats", user_stats_cmd))
    application.add_handler(CommandHandler("admin_stats", admin_stats))
    application.add_handler(CommandHandler("broadcast", admin_broadcast))
    application.add_handler(CommandHandler("ban", admin_ban))
    application.add_handler(CommandHandler("unban", admin_unban))

    # URL detection (must run before general text)
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_possible_url)
    )
    # Text‑to‑speech trigger (only if no URL was found)
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, text_message)
    )

    # Audio/Voice → transcription
    application.add_handler(
        MessageHandler(filters.VOICE | filters.AUDIO, handle_audio)
    )

    # Callback handlers – be precise with patterns to avoid conflicts
    application.add_handler(CallbackQueryHandler(
        callback_download_format, pattern="^dl:(bestvideo|1080|720|audio)$"
    ))
    application.add_handler(CallbackQueryHandler(
        search_callback, pattern="^dl:search:"
    ))
    application.add_handler(CallbackQueryHandler(
        playlist_callback, pattern="^pl:"
    ))
    application.add_handler(CallbackQueryHandler(
        tts_voice_callback, pattern="^tts:"
    ))
    application.add_handler(CallbackQueryHandler(
        inline_download_callback, pattern="^inline_dl:"
    ))

    # Trimming conversation (entry point: dl:trim)
    trim_conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(trim_start_prompt, pattern="^dl:trim$")],
        states={
            TRIM_START: [MessageHandler(filters.TEXT & ~filters.COMMAND, trim_start_input)],
            TRIM_END: [MessageHandler(filters.TEXT & ~filters.COMMAND, trim_end_input)],
        },
        fallbacks=[],
    )
    application.add_handler(trim_conv)

    # Inline handler
    application.add_handler(InlineQueryHandler(inline_query_handler))


async def post_init(application: Application) -> None:
    """Background tasks after the application is built."""
    asyncio.create_task(preload_whisper())


async def main_async() -> None:
    """Async entry point: init DB, build and run bot."""
    await init_db()
    application = Application.builder().token(BOT_TOKEN).build()
    application.post_init = post_init
    _setup_handlers(application)

    # Graceful shutdown via signal
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Shutdown signal received.")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Run polling until stop_event is set
    async with application:
        await application.start()
        # Wait for shutdown signal
        await stop_event.wait()
        await application.stop()


def main() -> None:
    """Synchronous entry point."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Bot interrupted by user.")
    finally:
        logger.info("Bot stopped.")


if __name__ == "__main__":
    main()
