from executorch.runtime import Runtime
import torch

# Assuming you have a pte file named 'model.pte'
pte_file_path = "weights/dfine_x_obj2coco.pte"

# 1. Load the PTE file data
with open(pte_file_path, "rb") as f:
    pte_data = f.read()

# 2. Load the program
runtime =Runtime.get()
handler = runtime.load_program(pte_data)

# 3. Load the specific method (e.g., "forward")
method = handler.load_method("forward")

# 4. Get the method metadata
method_meta = method.metadata

# Iterate over inputs and print their scalar types
for i in range(method_meta.num_inputs()):
    input_meta = method_meta.input_tensor_meta(i)
    if input_meta:
        # The scalar type is available as input_meta.scalar_type()
        # It's an enum, which you can typically compare or print.
        scalar_type = input_meta.dtype
        print(f"Input {i} scalar type: {scalar_type}")
        
        # You can compare it to torch.dtype values in the Python runtime
        # The exact mapping depends on the ExecuTorch version and mode (ATen/portable)
        # For example, to check if it's float:
        if scalar_type == torch.float32:
            print(f"Input {i} is float32")

# Iterate over outputs and print their scalar types
for i in range(method_meta.num_outputs()):
    output_meta = method_meta.output_tensor_meta(i)
    if output_meta:
        scalar_type = output_meta.dtype
        print(f"Output {i} scalar type: {scalar_type}")
