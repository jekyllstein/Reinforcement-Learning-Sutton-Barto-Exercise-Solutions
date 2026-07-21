# Reinforcement-Learning-Sutton-Barto-Exercise-Solutions

Chapter notes and exercise solutions for Reinforcement Learning: An Introduction, 2nd edition
by Richard S. Sutton and Andrew G. Barto.  The book in its entirety can be found here: http://www.incompleteideas.net/book/the-book-2nd.html

The notes are mostly in the format of [Pluto Notebooks](https://plutojl.org) which are interactive HTML notebooks that use the [Julia Programming Language](https://julialang.org).  If you follow the [installation instructions](https://julialang.org/downloads) to set up Julia on your system, then you can run the notebooks locally, use the interactive features, and edit them as desired.  Otherwise, there are [static HTML exports](https://jekyllstein.github.io/Reinforcement-Learning-Sutton-Barto-Exercise-Solutions/) of the notebooks.  See below for further details on accessing the static exports and running the notebooks locally.

## Videos from Online Meeting Discussing Notes

[Reinforcement Learning Tutorial Meetings - YouTube](https://youtube.com/playlist?list=PLYqXmZaxvwmy2CNaK-DLailou1VIU1UZn&si=bua2TF2lSU3KTxM6)

These videos were recorded during meetup group online meetings.  See the playlist and individual video descriptions for links to future meetings.

## View static exports of notebooks at https://jekyllstein.github.io/Reinforcement-Learning-Sutton-Barto-Exercise-Solutions/

If you view a notebook in your web browser, there will be instructions on how to download the individual file and run the notebook locally.

*Note: Not all notebooks are available here.  I am editing them to allow proper export and will be adding chapters one at a time.

## Running Notebooks Locally

To access all the notebooks there is a startup script that will finish instantiating the local packages and run Pluto in a temporary environment. Follow the instructions below:

1. Clone this repository
2. Open a terminal in the root folder of the repository and run the following command: 

```shell
./start.sh
```
  You may need to change the file permissions on `start.sh` so that it can run as an executable.  If you see a response like `permission denied: ./start.sh` then try running `chmod +x start.sh` to fix the problem.
  
  Note that this assumes that the julia command is in your environmental variables.  If not, then `julia` inside the shell script can be replaced with the path to the executable or whatever symbolic link name you are using to access your julia installation. If you follow the installation instructions on the Julia homepage then it should already be set up to run this way.

After executing `start.sh` you should see something that resembles the following in the terminal

```shell
[ Info: Loading...
┌ Info:
│ Go to http://localhost:1234/?secret=h8Ej8zIn in your browser to start writing ~ have fun!
└
┌ Info:
│ Press Ctrl+C in this terminal to stop Pluto
```

You can then paste the link into the browser of your choice and have access to a web interface that let's you select and run Pluto notebooks contained in the repository.  For further instructions on navigating the Pluto web interface, see the links above to resources on `pluto.jl`