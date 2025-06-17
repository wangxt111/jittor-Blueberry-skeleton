from datetime import datetime
import os

now = datetime.now().strftime("%Y%m%d_%H%M%S") 
default_output_dir = os.path.join('output', now, 'skeleton')

print(default_output_dir)
