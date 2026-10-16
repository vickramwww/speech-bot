import os
import asyncio
import base64
import json
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from azure.core.credentials import AzureKeyCredential
from azure.ai.voicelive.aio import connect
from azure.ai.voicelive.models import (
    RequestSession,
    ServerVad,
    AzureStandardVoice,
    Modality,
    InputAudioFormat,
    OutputAudioFormat,
    ServerEventType,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Browser connected")

    endpoint = os.environ.get("AZURE_VOICELIVE_ENDPOINT", "wss://api.voicelive.com/v1")
    api_key = os.environ.get("AZURE_VOICELIVE_API_KEY")
    model = os.environ.get("VOICELIVE_MODEL", "gpt-realtime")
    voice_name = os.environ.get("VOICELIVE_VOICE", "en-US-AvaNeural")
    instructions = os.environ.get(
        "VOICELIVE_INSTRUCTIONS",
        "You are a helpful AI assistant. Respond naturally and conversationally.",
    )

    if not api_key:
        await websocket.send_json({"type": "error", "message": "AZURE_VOICELIVE_API_KEY not configured on server"})
        await websocket.close()
        return

    try:
        async with connect(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key),
            model=model,
            connection_options={"max_msg_size": 1000 * 1024 * 1024, "heartbeat": 20, "timeout": 20},
        ) as vl_conn:

            voice_config = (
                AzureStandardVoice(name=voice_name, type="azure-standard") if "-" in voice_name else voice_name
            )
            await vl_conn.session.update(
                session=RequestSession(
                    modalities=[Modality.TEXT, Modality.AUDIO],
                    instructions=instructions,
                    voice=voice_config,
                    input_audio_format=InputAudioFormat.PCM16,
                    output_audio_format=OutputAudioFormat.PCM16,
                    turn_detection=ServerVad(threshold=0.5, prefix_padding_ms=300, silence_duration_ms=1000),
                )
            )

            await websocket.send_json({"type": "status", "message": "ready"})
            logger.info("VoiceLive session configured, ready")

            async def browser_to_voicelive():
                try:
                    while True:
                        raw = await websocket.receive_text()
                        msg = json.loads(raw)
                        if msg.get("type") == "audio":
                            await vl_conn.input_audio_buffer.append(audio=msg["data"])
                except (WebSocketDisconnect, Exception) as e:
                    logger.info(f"Browser disconnected from receive loop: {e}")

            async def voicelive_to_browser():
                try:
                    async for event in vl_conn:
                        if event.type == ServerEventType.RESPONSE_AUDIO_DELTA:
                            delta = event.delta
                            audio_b64 = delta if isinstance(delta, str) else base64.b64encode(bytes(delta)).decode()
                            await websocket.send_json({"type": "audio", "data": audio_b64})

                        elif event.type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED:
                            logger.info("Speech started")
                            await websocket.send_json({"type": "status", "message": "listening"})
                            try:
                                await vl_conn.response.cancel()
                            except Exception:
                                pass

                        elif event.type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STOPPED:
                            logger.info("Speech stopped")
                            await websocket.send_json({"type": "status", "message": "processing"})

                        elif event.type == ServerEventType.RESPONSE_AUDIO_DONE:
                            await websocket.send_json({"type": "status", "message": "speaking_done"})

                        elif event.type == ServerEventType.RESPONSE_DONE:
                            await websocket.send_json({"type": "status", "message": "ready"})

                        elif event.type == ServerEventType.ERROR:
                            logger.error(f"VoiceLive error: {event.error.message}")
                            await websocket.send_json({"type": "error", "message": event.error.message})

                except (WebSocketDisconnect, Exception) as e:
                    logger.info(f"VoiceLive event loop ended: {e}")

            await asyncio.gather(browser_to_voicelive(), voicelive_to_browser())

    except WebSocketDisconnect:
        logger.info("Browser disconnected")
    except Exception as e:
        logger.error(f"Session error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close()
        except Exception:
            pass
