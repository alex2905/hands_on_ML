# hands_on_ML
Repo for the book "Hands-On Machine Learning with Scikit-Learn, Keraos & TensorFlow"

## Windows bash, git, conda and make target setup

Follow the instructions here for the bash setup, using git, bash and conda:  
https://www.earthdatascience.org/workshops/setup-earth-analytics-python/setup-git-bash-conda/  

and for being able to use make targets in the git bash
https://gist.github.com/evanwill/0207876c3243bbb6863e65ec5dc3f058

adding Wget and make for terminal.

For initializing the git bash for conda run
```conda init bash```
in the anaconda prompt,  
After this is in place the following bash extensions can be installed if wished:  
```
cd ~  
git clone https://github.com/magicmonty/bash-git-prompt.git .bash-git-prompt --depth=1  
cat <<EOT >> ~/.bashrc  
GIT_PROMPT_ONLY_IN_REPO=1  
GIT_PROMPT_FETCH_REMOTE_STATUS=0  
. ~/.bash-git-prompt/gitprompt.sh  
EOT
```

```
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf  
~/.fzf/install

```

For creating the pthon environment open the git bash prompt in this repo and execute:  
```conda env create -n hands-on-ml --force --file environment.yml```
