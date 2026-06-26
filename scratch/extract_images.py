import json
import base64
import os

with open(r'e:\kuliah\AI PA3\training\evaluasi_indobert.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

img_count = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for output in cell.get('outputs', []):
            if 'data' in output and 'image/png' in output['data']:
                img_data = output['data']['image/png']
                if isinstance(img_data, list):
                    img_data = "".join(img_data)
                
                img_count += 1
                if img_count == 1:
                    fname = r'C:\Users\whisn\.gemini\antigravity-ide\brain\e403dbe2-2388-460c-b878-893751379c86\loss_curve.png'
                elif img_count == 2:
                    fname = r'C:\Users\whisn\.gemini\antigravity-ide\brain\e403dbe2-2388-460c-b878-893751379c86\confusion_matrix.png'
                else:
                    fname = rf'C:\Users\whisn\.gemini\antigravity-ide\brain\e403dbe2-2388-460c-b878-893751379c86\extra_img_{img_count}.png'
                
                with open(fname, "wb") as imgf:
                    imgf.write(base64.b64decode(img_data))
                print(f"Saved {fname}")
