# Nimble Miner

Program to mine tokens with Nimble Network

### Introduction

This repository is a subnet for text prompting with large language models (LLM). Inside, you will find miners designed by the Nimble Labs LTD team to serve language models. The current validator implementation queries the network for responses while servers responds to requests with their best completions. These completions are judged and ranked by the validators and passed to the chain.

### Install

This repository requires python3.8 or higher. To install, simply clone this repository and install the requirements.

```bash
git clone https://github.com/nimble-technology/nimble-miner.git
cd nimble-miner
python -m pip install -r requirements.txt
python -m pip install -e .
```

If you are running nimbleLM server, you might need install nimbleLM server requirements.

```bash
cd miners/nimbleLM/miner.py
python -m pip install -r requirements.txt
```

### Check wallet ballance
```bash
nbcli wallet overview --wallet.name miner
```

### Run the miner

Once you have done above steps, you can run the miner with the following commands.

```bash
# To run the miner
python -m miners/nimbleLM/miner.py
    --netuid 1
    --wallet.name <your miner wallet> # Must be created using the nimble-cli
    --wallet.hotkey <your miner hotkey> # Must be created using the nimble-cli
    --miner.blacklist.whitelist <hotkeys of the validators to be connected>
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
```

### Unstake from the hotkey to coldkey
```bash
nbcli stake remove --wallet.name miner
```
