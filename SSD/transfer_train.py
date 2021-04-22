import pathlib
import torch
import os
from ssd.config.defaults import cfg
from train import get_parser

if __name__ == "__main__":
   parser = get_parser()
   args = parser.parse_args()
   cfg.merge_from_file(args.config_file)
   cfg.merge_from_list(args.opts)
   output_dir = pathlib.Path(cfg.OUTPUT_DIR)
   # Define new output directory to
   new_output_dir = pathlib.Path(
       output_dir.parent,
       output_dir.stem + "_tdt4265"
   )

   # Copy checkpoint
   new_output_dir.mkdir()
   new_checkpoint_path = new_output_dir.joinpath("rdd2020_model.pth")
   previous_checkpoint_path = pathlib.Path(cfg.OUTPUT_DIR, "model_final.pth")
   assert previous_checkpoint_path.is_file()
   # Only keep the parameters for the model
   new_checkpoint = {
       "model": torch.load(previous_checkpoint_path)["model"]
   }
   torch.save(new_checkpoint, str(new_checkpoint_path))
   del new_checkpoint

   with open(args.config_file, "r") as fp:
       old_config_lines = fp.readlines()
   new_config_lines = []
   # Overwrite config values
   for line in old_config_lines:
       if line == '    TRAIN: ("rdd2020_train",)\n':
           old = line
           line = '    TRAIN: ("tdt4265_train",)\n'
           print(f"overwriting: {old} with {line}")
       if line == '    TRAIN: ("rdd2020_train_oversampling",)\n':
           old = line
           line = '    TRAIN: ("tdt4265_train",)\n'  # TODO Add oversamling for tdt dataset
           print(f"overwriting: {old} with {line}")
       if line == '    TEST: ("rdd2020_val", )\n':
           old = line
           line = '    TEST: ("tdt4265_val", )\n'
           print(f"overwriting: {old} with {line}")
       if line.startswith('OUTPUT_DIR:'):
           line = f'OUTPUT_DIR: {new_output_dir}\n'
       new_config_lines.append(line)
   new_config_path = pathlib.Path(args.config_file).parent.joinpath(new_output_dir.stem + ".yaml")
   with open(new_config_path, "w") as fp:
       fp.writelines(new_config_lines)
   print("New config saved to:", new_config_path)
   print("Starting train")
   os.system(f"python train.py {new_config_path}")