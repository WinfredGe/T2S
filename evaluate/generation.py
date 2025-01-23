import mgzip
import pickle
import os
from model.timevae_model import initialize_timevae_model
from .utils import show_with_start_divider,show_with_end_divider, make_sure_path_exist

def generate_data(cfg,data):
    show_with_start_divider(f"Data generation with settings:{cfg}")

    # Parse configs
    dataset_name = cfg.get('dataset_name','dataset')
    model_name = cfg.get('model','TimeVAE')
    output_gen_path = cfg.get('output_gen_path',r'./data/gen/')

    # Check dataset
    if data is None:
        show_with_end_divider('Error: Data for generation not found.')
        return None
    if isinstance(data, (list, tuple)) and len(data) == 2:
        train_data, valid_data = data
    else:
        show_with_end_divider('Error: Data for generation is invalid.')
        return None

    if model_name == 'TimeVAE':
        model = initialize_timevae_model(cfg,train_data[0].shape)
    else:
        show_with_end_divider('Error: Model not implemented.')
        return None

    if model is None:
        show_with_end_divider('Error: Model initialization failed.')
        return None
    
    generated_data = model.sample_data(train_data.shape[0])
    output_path = os.path.join(output_gen_path,model_name,f'{dataset_name}_gen.pkl')
    make_sure_path_exist(output_path)
    with mgzip.open(os.path.join(output_path), 'wb') as f:
        pickle.dump(train_data, f)
    show_with_end_divider(f'Generation done. Generated files saved to {output_path}.')
    return generated_data

def load_generated_data(cfg):
    show_with_start_divider(f"Load generated data with settings:{cfg}")

    # Parse configs
    dataset_name = cfg.get('dataset_name','dataset')
    model_name = cfg.get('model','TimeVAE')
    output_gen_path = cfg.get('output_gen_path',r'./data/gen/')

    file_path = os.path.join(output_gen_path,model_name)
    gen_data_path = os.path.join(file_path,f'{dataset_name}_train.pkl')

    # Read generated data
    if not os.path.exists(gen_data_path):
        show_with_end_divider(f'Error: Generated file in {file_path} does not exist.')
        return None
    try:
        with mgzip.open(gen_data_path, 'rb') as f:
            generated_data = pickle.load(f)
    except Exception as e:
        show_with_end_divider(f"Error: An error occurred during reading data: {e}.")
        return None

    show_with_end_divider(f'Generated data by model {model_name} loaded.')
    return generated_data