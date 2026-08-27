import json, glob, statistics

f = [x for x in glob.glob('/home/y50063564/DREAM-S/eval/data/MathVista*temp1.0*.jsonl')]
print('文件:', f)
if f:
    records = [json.loads(l) for l in open(f[0])]
    recs = records[-100:]
    n = len(recs)
    first50 = [float(r['average_accept_length']) for r in recs[:50]]
    last50 = [float(r['average_accept_length']) for r in recs[50:100]]
    print(f'总 {n} 条')
    print(f'前50: 平均 {statistics.mean(first50):.3f}')
    print(f'后50: 平均 {statistics.mean(last50):.3f}')
    print(f'全部: 平均 {statistics.mean(first50+last50):.3f}')
    # 对照采集的对应样本
    cap = [json.loads(l) for l in open('/home/y50063564/DREAM-S/eval/data/semantic_capture_shuffled.jsonl')]
    cap_accept_sum = sum(st.get('accept_length',0) for s in cap for st in s['steps'])
    cap_steps = sum(1 for s in cap for st in s['steps'])
    print(f'采集(50条): 每步accept均值 {cap_accept_sum/cap_steps:.3f} -> eval应报 {cap_accept_sum/cap_steps+1:.3f}')
