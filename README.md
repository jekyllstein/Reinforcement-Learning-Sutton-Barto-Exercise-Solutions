# Reinforcement-Learning-Sutton-Barto-Exercise-Solutions
Chapter notes and exercise solutions for Reinforcement Learning: An Introduction, 2nd edition
by Richard S. Sutton and Andrew G. Barto.  The book in its entirety can be found here: http://www.incompleteideas.net/book/the-book-2nd.html

The notes are mostly in the format of [Pluto Notebooks](https://plutojl.org) which are interactive HTML notebooks that use the [Julia Programming Language](https://julialang.org).  If you follow the [installation instructions](https://julialang.org/downloads) to set up Julia on your system, then you can run the notebooks locally, use the interactive features, and edit them as desired.  Otherwise, there are [static HTML exports](https://jekyllstein.github.io/Reinforcement-Learning-Sutton-Barto-Exercise-Solutions/) of the notebooks.  See below for further details on accessing the static exports and running the notebooks locally.

## View static exports of notebooks at https://jekyllstein.github.io/Reinforcement-Learning-Sutton-Barto-Exercise-Solutions/
If you view a notebook in your web browser, there will be instructions on how to download the individual file and run the notebook locally.

*Note: Not all notebooks are available here.  I am editing them to allow proper export and will be adding chapters one at a time.

## Running Notebooks Locally
To access all the notebooks there is a startup environment to conveniently set up Pluto and open a web browser to explore notebooks in any directory.  Follow the instructions below:

1. Clone this repository
2. Open a terminal in the root folder of the repository and run julia with the following command: 

```shell
julia --threads auto -e 'using Pkg; Pkg.a
ctivate("PlutoStartup"); Pkg.instantiate(); using PlutoStartup'
```
You can also run `./start.sh` which will execute the same command.  Note that this assumes that the julia command is in your environmental variables.  If not, then `julia` can be replaced with the path to the executable or whatever symbolic link name you are using to access your julia installation. If you follow the installation instructions on the Julia homepage then it should already be set up to run this way.