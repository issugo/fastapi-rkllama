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
