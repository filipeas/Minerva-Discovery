#!/bin/bash

echo "Setup and Run Project"
./setup_and_run.sh --container

# echo "🎈 Installing Minerva-Dev"
# (cd Minerva-Dev && pip install -e .)


echo "🎈 Everything installed"

# ------ Adding useful options --------
echo "🎈 Adding useful options...."

# Add tmux options
# Syncronize panes using Control+b e
# Disable syncronize panes using Control+b E
# Color support
# Mouse support
echo -e "set -g default-terminal \"screen-256color\"\nset -g mouse on\nbind e setw synchronize-panes on\nbind E setw synchronize-panes off" >> ~/.tmux.conf

# remove full path from prompt
sed -i '/^\s*PS1.*\\w/s/\\w/\\W/g' ~/.bashrc

# Setup the links to 
if [ -d "/workspaces/shared/data" ]; then
    ln -s "/workspaces/shared/data" shared_data
fi
if [ -d "/workspaces/shared/runs" ]; then
    ln -s "/workspaces/shared/runs" shared_runs
fi

echo "Done 🎉 🎈"
exit 0
