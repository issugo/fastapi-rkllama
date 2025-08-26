# fastapi-RKLLama: LLM Server and Client for Rockchip 3588/3576

## Project History

This project is a fork of [NotPunchnox/rkllama](https://github.com/NotPunchnox/rkllama) migrated to [fastAPI](https://fastapi.tiangolo.com/).

Big thanks to **NotPunchnox** to have initiated rkllama !

### Docker Installation

Pull the fastapi-RKLLama Docker image:

```bash
docker pull ghcr.io/issugo/fastapi-rkllama:latest
```
run server
```bash
docker run -it --privileged -p 8080:8080 ghcr.io/issugo/fastapi-rkllama:latest
```

### Models

You can find a good collection of models by visiting [dulimov huggingface profile](https://huggingface.co/dulimov).

Tested are Quen3 and Deepseek-R1-0528:
- [Qwen3-4B-Thinking-2507-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm](https://huggingface.co/dulimov/Qwen3-4B-Thinking-2507-rk3588-1.2.1/blob/main/Qwen3-4B-Thinking-2507-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm)
- [DeepSeek-R1-0528-Qwen3-8B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm](https://huggingface.co/dulimov/DeepSeek-R1-0528-Qwen3-8B-rk3588-1.2.1/blob/main/DeepSeek-R1-0528-Qwen3-8B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm)
- [Qwen3-8B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm](https://huggingface.co/dulimov/Qwen3-8B-rk3588-1.2.1-unsloth-16k/blob/main/Qwen3-8B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm)
- [Qwen3-4B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm](https://huggingface.co/dulimov/Qwen3-4B-rk3588-1.2.1-unsloth-16k/blob/main/Qwen3-4B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm)
- [Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm](https://huggingface.co/dulimov/Qwen3-1.7B-rk3588-1.2.1-unsloth-16k/blob/main/Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm)

## Contributing

This project is set up using [PyCharm](https://www.jetbrains.com/fr-fr/pycharm/).

Supported target are **Ubuntu 24** Armbian [OrangePI 5](https://www.armbian.com/orangepi-5/) or [raxda rock 5C](https://www.armbian.com/radxa-rock-5c/).

- Setup your SBC with Armbian
- create a default user
- on your dev computer, install PyCharm
- setup [Remote Development ...](https://www.jetbrains.com/help/pycharm/remote-development-starting-page.html) using ssh
- create [uv environment](https://www.jetbrains.com/help/pycharm/uv.html) (existing)

## Contributors
<a href="https://github.com/issugo/fastapi-rkllama/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=issugo/fastapi-rkllama" />
</a>
