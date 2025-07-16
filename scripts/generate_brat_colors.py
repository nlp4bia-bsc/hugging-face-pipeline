import colorsys
import random

"""
Script to generate distinctive colors for Brat (equally spaced in hue + randomnly 
variation of lightning and saturation for that hue, so they are more different).
You should include the output of this file to the <brat_project_root>/data/visual.conf
or <brat_project_root>/visual.conf.

Author: Jan Rodríguez
Date: 24/05/2023 v1.0
"""

# The list of all entities. Modify this to your case.
entities = [
    "Biomedical_Device_SCT",
    "Body_Tissue_SCT",
    "Cell_Sctructure_SCT",
    "Implant_SCT",
    "Material_SCT",
    "CHEMICAL",
    "GENE",
    "DEB_AdverseEffects",
    "DEB_ArchitecturalOrganization",
    "DEB_AssociatedBiologicalProcess",
    "DEB_BiologicallyActiveSubstance",
    "DEB_Biomaterial",
    "DEB_BiomaterialType",
    "DEB_Cell",
    "DEB_DegradationFeatures",
    "DEB_EffectOnBiologicalSystem",
    "DEB_ManufacturedObject",
    "DEB_ManufacturedObjectComponent",
    "DEB_ManufacturedObjectFeatures",
    "DEB_MaterialProcessing",
    "DEB_MedicalApplication",
    "DEB_ResearchTechnique",
    "DEB_Shape",
    "DEB_Species",
    "DEB_Structure",
    "DEB_StudyType",
    "DEB_Tissue",
    "SCI_AMINO_ACID",
    "SCI_ANATOMICAL_SYSTEM",
    "SCI_CANCER",
    "SCI_CELL",
    "SCI_CELL_LINE",
    "SCI_CELL_TYPE",
    "SCI_CELLULAR_COMPONENT",
    "SCI_CHEBI",
    "SCI_CHEMICAL",
    "SCI_CL",
    "SCI_DEVELOPING_ANATOMICAL_STRUCTURE",
    "SCI_DISEASE",
    "SCI_DNA",
    "SCI_GENE_OR_GENE_PRODUCT",
    "SCI_GGP",
    "SCI_GO",
    "SCI_IMMATERIAL_ANATOMICAL_ENTITY",
    "SCI_MULTI_TISSUE_STRUCTURE",
    "SCI_ORGAN",
    "SCI_ORGANISM",
    "SCI_ORGANISM_SUBDIVISION",
    "SCI_ORGANISM_SUBSTANCE",
    "SCI_PATHOLOGICAL_FORMATION",
    "SCI_PROTEIN",
    "SCI_RNA",
    "SCI_SIMPLE_CHEMICAL",
    "SCI_SO",
    "SCI_TAXON",
    "SCI_TISSUE"
]

num_colors = len(entities)
colors = []

# Generate the distinct colors using modified HSL values
for i in range(num_colors):
    hue = i * (360 / num_colors)
    # Random saturation between 40 and 70. Tune it if necessary.
    saturation = random.uniform(40, 70)
    # Random lightness between 20 and 60. Tune it if necessary.
    lightness = random.uniform(20, 60)
    hsl = (hue, saturation, lightness)
    rgb = colorsys.hls_to_rgb(hsl[0] / 360, hsl[2] / 100, hsl[1] / 100)
    colors.append(rgb)

# Randomize the association between entities and colors (this is to avoid contiguous entities in the list)
random.shuffle(colors)

# Print the colors in the desired format
for i in range(num_colors):
    entity = entities[i]
    color = colors[i]
    r, g, b = color
    rgb = f"RGB: {int(r * 255)}, {int(g * 255)}, {int(b * 255)}"
    bgColor = f"bgColor:#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
    fgColor = "fgColor:white" if r * 0.299 + g * 0.587 + b * 0.114 <= 0.5 else ""

    output = f"{entity} {bgColor}"
    if fgColor:
        output += f", {fgColor}"

    print(output)
