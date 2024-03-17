import collections
from typing import Optional

import numpy as np
import torch
from torch.nn import functional as F

NUM_DIMENSIONS = 3
Triplet = collections.namedtuple("Triplet", "a, b, c")

DEFAULT_OFFSETS = (1, 2, 3)
RINGER_OFFSETS = (0, 1, 1)
NUM_INTERNALS = 3

offsets = RINGER_OFFSETS


def extract_bd_theta_np(positions: np.ndarray):
    """Extracts the bond distance and angle in radians for a given tetrad structure in numpy array.

    Args:
        positions (np.ndarray): An array representing the positions of atoms in a tetrad structure.
            The structure is given by (a,b,c,d), where d is a branching atom.
            Dimensions: K x B x 4 x 3.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The calculated bond distance and angle in radians.
    """
    a = positions[:, :, 0, :]
    b = positions[:, :, 1, :]
    c = positions[:, :, 2, :]
    d = positions[:, :, 3, :]

    # calculate the main vectors
    ba = a - b
    bc = c - b
    bd = d - b

    cross_product = np.cross(bc, ba, axis=-1)

    norm = np.linalg.norm(cross_product, axis=-1, keepdims=False)  # Shape: (B, 1)
    bd_norm = np.linalg.norm(bd, axis=-1, keepdims=False)

    dot_prod = (cross_product * bd).sum(axis=-1)
    cos_angle = dot_prod / (norm * bd_norm)
    angle_radians = np.arccos(cos_angle)

    return bd_norm, angle_radians


def extract_bd_theta(positions: torch.FloatTensor):
    """Extracts the bond distance and angle in radians for a given tetrad structure in torch
    tensor.

    Args:
        positions (torch.FloatTensor): A tensor representing the positions of atoms in a tetrad structure.
            The structure is given by (a,b,c,d), where d is a branching atom.
            Dimensions: K x B x 4 x 3.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The calculated bond distance and angle in radians.
    """
    a = positions[:, :, 0, :]
    b = positions[:, :, 1, :]
    c = positions[:, :, 2, :]
    d = positions[:, :, 3, :]

    # calculate the main vectors
    ba = a - b
    bc = c - b
    bd = d - b

    cross_product = torch.cross(bc, ba, dim=-1)
    norm = torch.norm(cross_product, dim=-1, keepdim=False)
    bd_norm = torch.norm(bd, dim=-1, keepdim=False)

    # calculate the angle between the normal vector and bd vector
    dot_prod = (cross_product * bd).sum(dim=-1)
    cos_angle = dot_prod / (norm * bd_norm)
    angle_radians = torch.arccos(cos_angle)

    return bd_norm, angle_radians


class NeRF(object):
    """Natural Extension Reference Frame (NeRF).

    Constructs cartesian coordinates from internal coordinates. NeRF requires a dependency matrix
    that sequentially constructs the molecule. As such, it creates a set of distances, angles, and
    dihedrals, (the internal coordinates) that correspond to specific indices. Hence, for
    (1,2,3,4), atom_id 4 will be placed with: a distance using (3,4), and an angle about (2,3,4),
    and the torsion with (1,2,3,4) that collectively define position 4. This is different from
    RINGER, which for atom 2 would use bond distance (2,3), angle (1,2,3), and dihedral (1,2,3,4),
    since these all are focused on atom_id 2.
    """

    def __init__(self, float_type: torch.double):
        """Torch doubles as preferred for numerical stability, but can use floats for training if
        used in the loop."""
        self.float_type = float_type

        if float_type == torch.double:
            self.np_float_type = np.float64
        elif float_type == torch.float:
            self.np_float_type = np.float32

        self.init_matrix = np.array(
            [[-np.sqrt(1.0 / 2.0), np.sqrt(3.0 / 2.0), 0], [-np.sqrt(2.0), 0, 0], [0, 0, 0]],
            dtype=self.np_float_type,
        )

    @staticmethod
    def convert_to_points(bonds: torch.Tensor, angles: torch.Tensor, dihedrals: torch.Tensor):
        """
        Convert bonds, angles, and dihedrals to points.
        -----------
        bonds: torch.Tensor,
            bond lengths (distance in angstroms)
        angles: torch.Tensor,
            bond angles theta, in radians
        dihedrals: torch.Tensor,
            bond dihedrals, in radians.
        """
        r_cos_theta = bonds * torch.cos(torch.pi - angles)
        r_sin_theta = bonds * torch.sin(torch.pi - angles)

        points_x = r_cos_theta
        points_y = r_sin_theta * torch.cos(dihedrals)
        points_z = r_sin_theta * torch.sin(dihedrals)

        points = torch.stack([points_x, points_y, points_z])  # 3 x B x L
        return points.permute(2, 1, 0)  # L x B x 3

    @staticmethod
    def extend(point, last_three_points):
        """
        point: B x 3,
            The coordinates for a single position.
        last_three_points: Triple(NamedTuple),
            NamedTuple container for holding the last three coordinates.
        """
        bc = F.normalize(last_three_points.c - last_three_points.b, dim=-1)
        n = F.normalize(torch.cross(last_three_points.b - last_three_points.a, bc), dim=-1)
        m = torch.stack([bc, torch.cross(n, bc), n]).permute(1, 2, 0)

        point = point.unsqueeze(2)  # Expand from B x 3 to B x 3 x 1 to enable bmm

        return torch.bmm(m, point).squeeze() + last_three_points.c

    @staticmethod
    def validate_index(index: torch.LongTensor):
        """Perform a simple validation that the index can correctly satisfy Cartesian generation in
        sequential order.

        If returns True, then we satisfy the correct traversal over the indices.
        """
        index = index.clone()
        new_index = torch.zeros(index.max() + NUM_DIMENSIONS + 1).type(torch.LongTensor)

        # We choose an arbitrary starting point set the last three indices as the new_index
        for i in range(3):
            index[i][: 3 - i] = torch.arange(i - 3, 0)

        # Because these are virtual nodes, they don't matter and we set these to True
        new_index[-3:] = 1

        # iterate simply to make sure things are zero
        for i in index:
            # assume if the previous three are set, then we can set the fourth
            if torch.all(new_index[i[:3]]):
                new_index[i[3]] = 1
            else:
                print(f"index failed for {i}")
                return False
        return True

    @staticmethod
    def build_indices(length, size, offset=0):
        a, b = torch.meshgrid(torch.arange(size), torch.arange(length), indexing="xy")
        index = (a + b - offset) % length
        return index

    def nerf(
        self,
        r_theta_phi: torch.Tensor,
        quadruples: Optional[torch.LongTensor] = None,
        validate_index: bool = True,
    ) -> torch.Tensor:
        """
        Apply the Natural extension Reference Frame Method to a set of internal coords.
        ----------
        Args:
        r_theta_phi: torch.FloatTensor,
            B x L x 3 Tensor of distances (r), angles (theta), and dihedrals (phi).
        quandruples: torch.LongTensor,
            A tensor of indices corresponding to each set of internal_coordinates.
        Returns:
        xyzs: torch.Tensor,
            B x L x 3 Tensor of x,y,z coordinates
        """
        if quadruples is None:
            quadruples = self.build_indices(r_theta_phi.size(1), size=4)  # get quadruples
        # Clone the indices
        if validate_index:
            self.validate_index(quadruples)

        quadruples = quadruples.clone()

        atom_indices = quadruples[
            :, -1
        ]  # the last column of the B x 4 index corresponds to the atom_ids

        # This creates three virtual atoms that overwrite existing ones for the start index
        for i in range(3):
            quadruples[i][: 3 - i] = torch.arange(i - 3, 0)

        batch_size = r_theta_phi.shape[0]
        points = self.convert_to_points(
            r_theta_phi[:, :, 0], r_theta_phi[:, :, 1], r_theta_phi[:, :, 2]
        )

        # we create a new index to index into
        new_index = torch.zeros(
            quadruples.max() + NUM_DIMENSIONS + 1, batch_size, NUM_DIMENSIONS
        ).type(torch.DoubleTensor)
        # index those points into the new_index to be looked up by the last col
        new_index[quadruples[:, -1]] = points

        init_tensor = torch.from_numpy(self.init_matrix)
        new_index[-3:, :, :] = torch.cat(
            [row.repeat(1).view(1, 3) for row in init_tensor]
        ).unsqueeze(1)

        coords_list = new_index.clone()
        for quadruple in quadruples:
            prev_three_coords = Triplet(
                coords_list[quadruple[0]], coords_list[quadruple[1]], coords_list[quadruple[2]]
            )
            coords_list[quadruple[3]] = self.extend(coords_list[quadruple[3]], prev_three_coords)

        # return just the columns corresponding to the atom_indices
        return (coords_list.permute(1, 0, 2))[:, atom_indices]

    def __call__(self, r_theta_phi: torch.Tensor, quadruples: Optional[torch.Tensor] = None):
        """Call NeRF to perform the reconstruction."""
        return self.nerf(r_theta_phi, quadruples)


class TetraPlacer:
    def calculate_bd(self, positions: torch.Tensor, bd_norm: torch.Tensor, theta: torch.Tensor):
        """Calculates the branched atom distance for a given structure.

        Args:
            positions (torch.Tensor): A tensor representing the positions of atoms.
                Dimensions: K x B x 3 x 3, with K as the batch size, B the number of structures in the batch, and 3x3 as the position vectors.
            bd_norm (torch.Tensor): A tensor representing the norms corresponding to the branched atom distance. Vector based, size N x 1.
            theta (torch.Tensor): A tensor representing the angle in degrees. Vector based, size N x 1.

        Returns:
            torch.Tensor: The calculated branched atom distance tensor.
        """
        a = positions[:, :, 0, :]
        b = positions[:, :, 1, :]
        c = positions[:, :, 2, :]

        # calculate the main vectors
        ba = a - b
        bc = c - b

        centroid = torch.mean(torch.stack([a, c]), dim=0)
        centroid_b = b - centroid  #

        a_prime = a + centroid_b
        c_prime = c + centroid_b

        cross_product = torch.cross(bc, ba, dim=-1)
        norm = torch.norm(cross_product, dim=-1, keepdim=True)

        normalized_cross_product = (cross_product / norm) * bd_norm.unsqueeze(-1) + b

        new_bd = self.rotate_ac(a_prime, normalized_cross_product, c_prime, theta, b_only=True)

        return new_bd

    def rotate_ac(self, a, b, c, theta, b_only: bool = True):
        """Performs batch rotation for the tensor, withholding K x B x 3 tensors.

        Args:
            a, b, c (torch.Tensor): Input tensors of dimensions K x B x 3, representing initial position vectors for each batch.
            theta (torch.Tensor): The angle of rotation in radians. Vector based, size N x 1.
            b_only (bool): If True, only returns tensor b_rot, otherwise also returns a and c tensors. Default is True.

        Returns:
            torch.Tensor: The rotated tensor b_rot or (if b_only=False) a, b_rot, c.
        """
        K, B, _ = a.shape

        # Expand theta for compatibility with K and B dimensions
        theta = theta.unsqueeze(-1)  # Now shape (K, B, 1) for compatibility with broadcasting

        # Calculate normalized ac direction
        ac = c - a
        ac_norm = ac / torch.norm(ac, dim=2, keepdim=True)

        zeros = torch.zeros_like(theta)
        k = ac_norm

        K_matrix = torch.cat(
            [
                zeros,
                -k[..., 2:3],
                k[..., 1:2],
                k[..., 2:3],
                zeros,
                -k[..., 0:1],
                -k[..., 1:2],
                k[..., 0:1],
                zeros,
            ],
            dim=2,
        ).reshape(
            K * B, 3, 3
        )  # Reshape for batch matrix multiplication

        # Calculate rotation matrix R and the three terms
        R_1 = torch.eye(3, device=k.device).repeat(K * B, 1, 1)  # Repeat eye for each batch
        R_2 = torch.sin(theta).reshape(K * B, 1, 1) * K_matrix
        R_3 = (1 - torch.cos(theta).reshape(K * B, 1, 1)) * torch.bmm(K_matrix, K_matrix)
        R = R_1 + R_2 + R_3

        # Translate b to the origin (relative to a), then rotate, then translate back
        b_translated = (b - a).reshape(K * B, 3, 1)  # Reshape for batch matrix multiplication
        b_rot = torch.bmm(R, b_translated).reshape(K, B, 3) + a
        if b_only:
            return b_rot
        return a, b_rot, c  # returns a and as a sanity check

    def add_branched_points(
        self,
        xyzs,
        quad_indices: torch.LongTensor,
        quad_bond_distances: torch.FloatTensor,
        quad_bond_thetas: torch.FloatTensor,
        copy: bool = True,
    ):
        """Adds a new branched point for each structure in the batch.

        Args:
            xyzs (torch.Tensor): The current tensor of atom positions in each structure, with dimensions K x M x 3.
            quad_indices (torch.FloatTensor): The indices of quads in the structures, with dimensions N x 4.
            quad_bond_distances (torch.FloatTensor): The distances of quad bonds, with dimensions N x 1.
            quad_bond_thetas (torch.FloatTensor): Theta values for each quad bond, with dimensions N x 1.
            copy (bool): If True, creates a copy of `xyzs` to perform operations on, else operates in-place. Default is True.

        Returns:
            torch.Tensor: The tensor of atom positions after adding the branched points.
        """
        if copy:
            xyzs = xyzs.clone()
        triples = quad_indices[:, :3]
        target_index = quad_indices[:, -1]

        positions = xyzs[:, triples, :]

        k = xyzs.shape[0]
        quad_bd = quad_bond_distances.repeat((k, 1))
        quad_theta = quad_bond_thetas.repeat((k, 1))

        # print(theta_mean.shape)
        target_xyz = self.calculate_bd(positions, quad_bd, quad_theta)
        xyzs[:, target_index, :] = target_xyz

        return xyzs


class RigidTransform(object):
    """Implementation of Kabsch algorithm in PyTorch to handle batching.

    Does not handle reflections at the moment.
    """

    def __init__(self):
        """R is the rotation matrix, t is the translation matrix."""
        self.R = None
        self.t = None

    def fit(self, source: torch.Tensor, target: torch.Tensor) -> None:
        assert source.shape == target.shape

        # find mean row wise
        centroid_source = torch.mean(source, dim=1)
        centroid_target = torch.mean(target, dim=1)

        # Center the data along the centroid target
        source_m = source - centroid_target.unsqueeze(dim=1)
        target_m = target - centroid_target.unsqueeze(dim=1)

        H = torch.bmm(source_m.permute(0, 2, 1), target_m)
        U, S, Vt = torch.linalg.svd(H)

        R = torch.bmm(Vt.permute(0, 2, 1), U.permute(0, 2, 1))
        t = -R @ centroid_source.unsqueeze(dim=2) + centroid_target.unsqueeze(dim=2)

        # Store these
        self.R = R
        self.t = t

    @staticmethod
    def get_reflections(R):
        """Not implemented at the moment."""
        return torch.where(torch.linalg.det(R) < 0)

    def transform(self, source):
        return torch.bmm(source, self.R.permute(0, 2, 1)) + self.t.permute(0, 2, 1)

    def fit_transform(self, source: torch.Tensor, target: torch.Tensor):
        """Fit and transform the data."""
        self.fit(source, target)
        return self.transform(source)

    @staticmethod
    def rmsd(source_tensor, target_tensor):
        """Calculate the RMSD between the source and the target tensors."""
        assert source_tensor.shape[1] == target_tensor.shape[1]
        return (
            ((source_tensor - target_tensor) ** 2).sum(-1).sum(-1) / source_tensor.shape[1]
        ) ** 0.5

    def __call__(self, source: torch.Tensor):
        transformed = self.transform(source)
        return transformed


class InverseNeRF(object):
    """Construct the internal coordinates, r_theta_phi, matrix from xyz coordinates.

    Currently only works for backbones.
    """

    def __init__(
        self,
        distances_offset: int = 1,  # 0 1
        angles_offset: int = 2,  # 1 1
        dihedrals_offset: int = 3,
    ):  # 1 2
        """
        distances_offset: int = 1,
            the distance from the previous atom_id to the current one.
        angles_offset: int = 2,
            the angle defining the atom, starting two atoms prior.
        dihedrals_offset: int = 3,
            get the dihedral corresponding to the atom two positions prior,
            which requires finding the start of the quadruple 3 points prior.
        """
        self.distances_offset = distances_offset
        self.angles_offset = angles_offset
        self.dihedrals_offset = dihedrals_offset

    @staticmethod
    def build_indices(length, size, offset: int = 0) -> torch.Tensor:
        """Build wrapped indices for cycles only."""
        a, b = torch.meshgrid(torch.arange(size), torch.arange(length), indexing="xy")
        index = (a + b - offset) % length
        return index

    def distances(self, positions, index: Optional[torch.Tensor] = None):
        """Calculate the distances with tuples, starting with the offset used."""
        if index is None:
            index = self.build_indices(positions.size(1), 2, self.distances_offset)
        doubles = positions[:, index, :]

        return torch.norm(doubles[..., 1, :] - doubles[..., 0, :], dim=-1)

    def angles(self, positions, index: Optional[torch.Tensor] = None):
        """Calculate bond angles."""
        if index is None:
            index = self.build_indices(positions.size(1), 3, self.angles_offset)
        triples = positions[:, index, :]

        a1 = triples[..., 1, :] - triples[..., 0, :]
        a2 = triples[..., 2, :] - triples[..., 1, :]

        a1 = F.normalize(a1, dim=-1)
        a2 = F.normalize(a2, dim=-1)

        rad = torch.pi - torch.arccos((a1 * a2).sum(dim=-1))

        return rad

    def dihedrals(self, positions, index: Optional[torch.Tensor] = None):
        """Calculate dihedral angles from sets of quadruples."""
        if index is None:
            index = self.build_indices(
                positions.size(1), 4, self.dihedrals_offset  # dihedrals require 4 indices
            )
        quadruples = positions[:, index, :]

        a1 = quadruples[..., 1, :] - quadruples[..., 0, :]
        a2 = quadruples[..., 2, :] - quadruples[..., 1, :]
        a3 = quadruples[..., 3, :] - quadruples[..., 2, :]

        v1 = torch.cross(a1, a2, dim=-1)
        v1 = F.normalize(v1, dim=-1)
        v2 = torch.cross(a2, a3, dim=-1)
        v2 = F.normalize(v2, dim=-1)

        sign = torch.sign((v1 * a3).sum(dim=-1))
        rad = torch.arccos((v1 * v2).sum(-1) / ((v1**2).sum(-1) * (v2**2).sum(-1)) ** 0.5)

        rad = sign * rad
        return rad

    def inverse_nerf(self, positions):
        """Collect distances, angles, and diehdrals to generate r_theta_phi matrix.

        Params:
        positions: torch.FloatTensor,
            N x L x 3 matrix of xyz coordinates (positions).
        Returns:
            N x L x 3 matrix of r, theta, and phis.
        """
        r = self.distances(positions)
        theta = self.angles(positions)
        phi = self.dihedrals(positions)
        r_theta_phi = torch.stack([r, theta, phi]).permute(1, 2, 0)
        return r_theta_phi

    def reindex(self, array, offset_differences, length: Optional[int] = None):
        """Reindex the array based on the differences in offsets."""
        if length is None:
            length = array.size(1)
        new_index = self.build_reindex(offset_differences, length)
        return array[:, new_index, torch.arange(array.size(-1))[None, :]]

    def build_reindex(self, offsets, length):
        reindex = (torch.arange(length).reshape(-1, 1) + np.array(offsets)) % length
        return reindex

    @staticmethod
    def convert_offsets(source_offset, target_offset):
        """Convert the difference in offsets.

        b
        """
        return tuple(source_offset[i] - target_offset[i] for i in range(NUM_INTERNALS))

    def __call__(self, positions):
        return self.inverse_nerf(positions)
