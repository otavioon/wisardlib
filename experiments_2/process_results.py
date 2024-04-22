import argparse
from pathlib import Path
import pandas as pd
import json

def main(results_dir: Path):
    json_files = list(results_dir.rglob('result.json'))
    print(f"There is {len(json_files)} json files")
    results = []
    for json_file in json_files:
        print(f"Processing: {json_file}")
        raw_data = json_file.open("r").read()
        
        if len(raw_data) < 10:
            print(f"\tFile {json_file} is not valid or empty")
            continue
    
        if raw_data.find("}\n{") != -1:
            raw_data = raw_data.split("}\n{")[-1]
            raw_data = "{" + raw_data
        
        data = json.loads(raw_data)
        # data["file"] = str(json_file)
        
        if 'config' in data:
            config = data.pop('config')
            data.update(config)
            
        data["ram"] =  str(json_file).split("/")[-2].split("Experiment")[0]
        
        results.append(data)
        # break
        
    pd.DataFrame(results).to_csv(results_dir / "results.csv", index=False)
    print(f"Total results: {len(results)}. Total files: {len(json_files)}")
    print("Done")
    
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, required=True)
    args = parser.parse_args()
    
    main(Path(args.results_dir))