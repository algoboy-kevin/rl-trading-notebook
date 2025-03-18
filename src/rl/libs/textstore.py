import os

def saveFloatData(value: float):
    file_path = './export/fitness_data.txt'
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Open the file in append mode, creating it if it doesn't exist
    with open(file_path, 'a+') as f:
        # Move to the beginning of the file
        f.seek(0)
        
        # Check if the file is empty
        if f.read(1):
            # If not empty, add a space before the new value
            f.write(' ')
        
        # Move to the end of the file and write the new value
        f.seek(0, 2)
        f.write(f"{value}")

def readFloatData() -> list[float]:
    file_path = './export/fitness_data.txt'
    
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, 'r') as f:
        content = f.read().strip()
        if not content:
            return []
        
        float_data = []
        for x in content.split():
            try:
                float_data.append(float(x))
            except ValueError:
                # Skip any string that can't be converted to float
                continue
        
        return float_data
