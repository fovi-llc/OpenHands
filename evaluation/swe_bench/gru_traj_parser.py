import json

from pyperclip import paste

try:
    traj = paste()
    traj = json.loads(traj)
    with open('evaluation/swe_bench/gru_traj.json', 'w') as f:
        f.write(traj)
except Exception:
    with open('evaluation/swe_bench/gru_traj.json', 'r') as f:
        traj = json.load(f)
# // script to print chat human friendly
for i in traj['trajectory']['details'][0]['exportedConversation']['plans'][0]:
    for j in i:
        print(j['thought'])
        print(j['name'], j['args'])
        print(j['observation'])
        print('--' * 100)
