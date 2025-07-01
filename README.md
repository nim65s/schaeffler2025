# Formation Schaeffler, 2025

This repository contains the exercices for the training session with Schaeffler, 2025.
The exercices are organized by notebook. Each notebook corresponds to one session of the class.
The notebooks are in Python and based on the software [Pinocchio](https://github.com/stack-of-tasks/pinocchio).

## Set up

```bash
uv run jupyter lab
```

## Use

### Update the notebooks

If the repository changes (for example when new tutorials are pushes), you need to update your local
version by "pulling" it from the repository.
On a native installation, just go in the folder containing the tutorials and execute `git pull`

To avoid conflict when pulling a new version, you should better do your modifications in copy of the original files,
not directly in the original files itself.
We then strongly suggest that you don't work on the notebooks directly, but on a copy of it (save 0-introduction_to_numerical_robotics.ipynb into something like 0_mycopy.ipynb before working on it).
