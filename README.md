# Formation Schaeffler, 2025

This repository contains the exercices for the training session with Schaeffler, 2025.
The exercices are organized by notebook. Each notebook corresponds to one session of the class.
The notebooks are in Python and based on the software [Pinocchio](https://github.com/stack-of-tasks/pinocchio).

## Set up

### Default setup

This setup should work on Linux / MacOS / WSL2.

You **must** use a [virtualenv](https://docs.python.org/3/library/venv.html) or a similar.

For example on Debian/Ubuntu, you can get started with:
```bash
sudo apt install python-is-python3 python3-pip python3-venv
```

To create a virtual environment and activate it:
```bash
python -m venv .venv
source .venv/bin/activate
```

**Once in your virtual environment**, install this package:
```bash
pip install -U pip
pip install .
```

After that, you can start the server with `jupyter lab`

### uv

Instead of using pip and venv, you can use this project directly with [uv](https://docs.astral.sh/uv/), which will
setup everything for you, and you'll only need one command:

```bash
uv run jupyter lab
```

### Docker

If the default setup is not working for you, as a backup solution, a Docker image is provided, and can be started with:

```bash
docker run --rm -p 7000:7000 -p 7001:7001 -p 7002:7002 -p 7003:7003 -p 7004:7004 -p 8888:8888 -v data:/home/user/tp -it gepetto/schaeffler2025
```

On Linux host systems, you may simply start the Docker with:

```bash
docker run --rm --net host -v data:/home/user/tp -it gepetto/schaeffler2025
```

In case of big update, you must update the docker:
```bash
docker pull gepetto/schaeffler2025
```

## Use

### Update the notebooks

If the repository changes (for example when new tutorials are pushes), you need to update your local
version by "pulling" it from the repository.
On a native installation, just go in the folder containing the tutorials and execute `git pull`


With a docker, execute the following:
```bash
docker run --rm -v data:/home/user/tp -it gepetto/schaeffler2025 git pull --rebase origin main
```

To avoid conflict when pulling a new version, you should better do your modifications in copy of the original files,
not directly in the original files itself.
We then strongly suggest that you don't work on the notebooks directly, but on a copy of it (save 0-introduction_to_numerical_robotics.ipynb into something like 0_mycopy.ipynb before working on it).

## Side notes

### Installing docker

See [how to get Docker](https://docs.docker.com/get-docker/).

## Docker as normal user

To avoid the need to run `sudo docker`:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

