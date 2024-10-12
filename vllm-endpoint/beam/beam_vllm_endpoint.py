from beam import asgi, Image, Volume

VOLUME_PATH = "./model"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
vllm_cache = Volume(name="vllm-weights", mount_path=VOLUME_PATH)

@asgi(
    name="vllm-openai-server",
    image=Image(python_version="python3.11")
    .add_commands(["apt-get update -y"])
    .add_commands([
        "pip install torch torchvision torchaudio",
        "pip install pydantic==2.9.2 vllm==0.6.2",
        "pip install --upgrade pyzmq fastapi uvicorn starlette"
    ]),
    cpu=2,
    gpu="A100-40",
    memory="16Gi",
    volumes=[vllm_cache],
    secrets=["HUGGING_FACE_HUB_TOKEN"],
    timeout=600
)
def serve():
    import asyncio
    import uvloop
    import os
    import logging
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.entrypoints.logger import RequestLogger
    from vllm.entrypoints.openai.api_server import (
        build_app,
        init_app_state,
    )
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils import FlexibleArgumentParser
    from vllm.usage.usage_lib import UsageContext

    # Set up the event loop policy
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Create the argument parser
    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)

    # Parse the default arguments
    args = parser.parse_args([])

    # Override specific arguments
    args.model = MODEL_NAME
    args.download_dir = os.path.abspath(VOLUME_PATH)
    args.max_model_len = 8192
    args.max_seq_len_to_capture = 16384
    args.chat_template = os.path.abspath("./template.jinja")
    args.gpu_memory_utilization = 0.90
    args.tensor_parallel_size = 1

    # Additional arguments required by init_app_state
    args.served_model_name = [MODEL_NAME]
    args.disable_log_requests = False
    args.max_log_len = 2048
    args.disable_log_stats = False
    args.response_role = "assistant"
    args.lora_modules = []
    args.prompt_adapters = []
    args.return_tokens_as_token_ids = False
    args.enable_auto_tool_choice = False
    args.tool_call_parser = None

    # Create engine arguments
    engine_args = AsyncEngineArgs.from_cli_args(args)

    # Initialize the engine client
    async_engine_client = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER
    )
    model_config = asyncio.run(async_engine_client.get_model_config())

    # Build the application
    app = build_app(args)

    # Initialize app state
    init_app_state(
        engine_client=async_engine_client,
        model_config=model_config,
        state=app.state,
        args=args,
    )

    return app
