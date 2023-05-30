from . import macrocycle

DATASET_CLASSES = {
    "distances-angles": macrocycle.MacrocycleInternalCoordinateDataset,
    "angles": macrocycle.MacrocycleAnglesDataset,
    "dihedrals": macrocycle.MacrocycleDihedralsDataset,
}
