
import json
import os
            
def write_to_json_file(file_path,data):
    if not os.path.exists(file_path):
        raise ValueError(f"The file {file_path} does not exist")
    file = None  
    try:
        file = open(file_path, 'w') 
        json.dump(data,file,indent=4) #convert json to string and write to file by tab space   
    except OSError as e:
        raise ValueError(f"Error opening file {file_path}: {e}")
    finally:
        if file: 
            file.close() 

def read_file_lines(file_path):
    input_file=None
    if not os.path.exists(file_path):
        raise ValueError(f"the file {file_path}  does not exist")  
    try:
        input_file=open(file_path, 'r' )
        lines=input_file.readlines()
        for i,line in  enumerate(lines):
            lines[i]=lines[i].strip()
        return lines
    except IOError as e:
        raise ValueError(f"an error occurred when reading file {file_path} : {e}")
    finally:
        if(input_file):
            input_file.close()

def read_file(file_path):
    
    if not os.path.exists(file_path):
        raise ValueError(f"the file {file_path}  does not exist")  
    input_file=None
    try:
        input_file=open(file_path, 'r' )
        return input_file.read().strip()
    except IOError as e:
        print(f"an error occurred when reading file {file_path}")
        print(e)
    finally:
        if(input_file):
            input_file.close()

def write_file(file_path,content):
    output_file=None
    try:
        output_file=open(file_path, 'w' )
        output_file.write(content+"\n")
    except IOError as e:
        raise ValueError(f"an error occurred when append file {file_path} : {e}")
    finally:
        if(output_file):
            output_file.close()
            
def append_file(file_path,content):
    if not os.path.exists(file_path):
        raise ValueError(f"the file {file_path}  does not exist")  
    output_file=None
    try:
        output_file=open(file_path, 'a' )
        output_file.write(content+"\n") 
    except IOError as e :
        raise ValueError(f"an error occurred when append file {file_path} : {e}")
    finally:
        if(output_file is not None):
            output_file.close()
