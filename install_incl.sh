# Add gcloud to the path
echo "source /google-cloud-sdk/path.zsh.inc" >> ${HOME}/.zshrc
echo "source /google-cloud-sdk/completion.zsh.inc" >> ${HOME}/.zshrc
echo "source /google-cloud-sdk/path.bash.inc" >> ${HOME}/.bashrc
echo "source /google-cloud-sdk/completion.bash.inc" >> ${HOME}/.bashrc

# Install INCL
git clone https://github.com/lunit-io/incl-client.git --depth=1 /tmp/incl-client &&
    pip install /tmp/incl-client &&
    rm -rf /tmp/incl-client