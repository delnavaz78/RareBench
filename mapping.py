import json

def map_id_to_name(json_file):
    """
    Extracts IDs  and their corresponding names frrom source file
    
    Args:
        json_file (str): Path to the JSON file.

    Returns:
        dict: A dictionary mapping {phenotype_id: phenotype_name}.
    """
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    disease_dict = {}

    # Navigate to the nodes section
    for graph in data.get("graphs", []):
        for node in graph.get("nodes", []):
            disease_id = node.get("id")  # Example: "http://purl.obolibrary.org/obo/HP_0000001"
            disease_id = "<" + disease_id + ">"
            disease_name = node.get("lbl")  # Example: "All"

            if disease_id and disease_name:
                disease_dict[disease_id] = disease_name

    return disease_dict

disease_map = map_id_to_name("data/mondo.json")
phen_map = map_id_to_name("data/hp.json")


# Save the phenotype map
with open("mapping/phen_map.json", "w", encoding="utf-8") as output_file:
    json.dump(phen_map, output_file, indent=4, ensure_ascii=False)

# Save the disease map
with open("mapping/disease_map.json", "w", encoding="utf-8") as output_file:
    json.dump(disease_map, output_file, indent=4, ensure_ascii=False)


id2name_map = disease_map.copy() 

for id, name in phen_map.items():
    if id not in id2name_map:
        id2name_map[id] = name

# Save the id2name map
with open("mapping/id2name_map(1).json", "w", encoding="utf-8") as output_file:
    json.dump(id2name_map, output_file, indent=4, ensure_ascii=False)
