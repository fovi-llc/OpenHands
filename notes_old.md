export OPENAI_API_KEY=sk-YOUR-API-KEY-HERE
export WORKSPACE_BASE=`pwd`/workspace

docker run \
    -it \
    -e SANDBOX_USER_ID=$(id -u) \
    -e WORKSPACE_MOUNT_PATH=$WORKSPACE_BASE \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    -v $WORKSPACE_BASE:/opt/workspace_base \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    ghcr.io/opendevin/opendevin:main


root:/app# mkdir -p /home/enduser/.cache/huggingface/hub

root:/app# source .venv/bin/activate

(opendevin-py3.12) root:/app# playwright install

# The -m is not really needed here because gpt-3.5-turbo is the default
(opendevin-py3.12) root:/app# python ./opendevin/core/main.py \
        -i 10 \
        -t "Write me a bash script that print hello world." \
        -c CodeActAgent \
        -m gpt-3.5-turbo


playwright._impl._api_types.Error: Executable doesn't exist at /root/.cache/ms-playwright/chromium-1084/chrome-linux/chrome
╔════════════════════════════════════════════════════════════╗
║ Looks like Playwright was just installed or updated.       ║
║ Please run the following command to download new browsers: ║
║                                                            ║
║     playwright install                                     ║
║                                                            ║
║ <3 Playwright Team                                         ║
╚════════════════════════════════════════════════════════════╝

ERROR:root:<class 'opendevin.runtime.browser.browser_env.BrowserException'>: Failed to start browser environment.

(opendevin-py3.12) root@345ab6fa9488:/app# playwright install


/app/.venv/lib/python3.12/site-packages/llama_index_client/types/metadata_filter.py:20: SyntaxWarning: invalid escape sequence '\*'
  """
There was a problem when trying to write in your cache folder (/home/enduser/.cache/huggingface/hub). You should set the environment variable TRANSFORMERS_CACHE to a writable directory.



python ./opendevin/core/main.py \
        -i 10 \
        -t "Change my Hello World script to print in several different languages." \
        -c CodeActAgent \
        -m gpt-4o


python ./opendevin/core/main.py         -i 10         -t "Approximate a 747 as a hollow aluminum tube with closed 
ends and appropriate dimensions and weight.  Calculate how far it will penetrate the surface of a level body of water if falling directly down, end first at terminal velocity.  Use your extensive knowledge of physics, fluid dynamics, and Python programming to use the appropriate formulas and code them in Python to do the calculation."         -c MathAgent -m gpt-4o

