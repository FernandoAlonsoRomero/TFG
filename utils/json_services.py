import json

def write_json(data_dict, path):
    """ Writes a new JSON file
        
        @type  feature_vector: dict.
        @param feature_vector: data which we want to save in the json file.
        
        @type  root_path: string.
        @param root_path: directory where you want to save the JSON file.
        
        @type  file_name: string.
        @param file_name: name of the new JSON file.
    """

    with open(path, "w") as outfile:
        json.dump(data_dict, outfile)
        
        
def load_json(root_path):
    """ Loads an existing JSON file and return the data

        @type  root_path: string.
        @param root_path: path of the JSON file.
        
        @rtype: dict
        @return: the data inside the JSON file
    """
    
    with open(root_path, "r") as json_file:
        data = json.load(json_file)

    return data
