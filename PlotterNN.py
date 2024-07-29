import os
import numpy as np
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
def mirror_mesh(nodes, elements,contour):
    num_original_nodes = nodes.shape[0]

    # Extract contour values
    contour_values = contour

    # Mirroring nodes along x=1 and y=1
    mirror_x = np.copy(nodes)
    mirror_x[:, 1] = 2 - mirror_x[:, 1]  # Mirror along x=1

    mirror_y = np.copy(nodes)
    mirror_y[:, 2] = 2 - mirror_y[:, 2]  # Mirror along y=1

    mirror_xy = np.copy(nodes)
    mirror_xy[:, 1] = 2 - mirror_xy[:, 1]
    mirror_xy[:, 2] = 2 - mirror_xy[:, 2]  # Mirror along both x=1 and y=1

    # Combine all nodes and duplicate contour values
    combined_nodes = np.vstack([nodes, mirror_x, mirror_y, mirror_xy])
    combined_contour_values = np.concatenate([contour_values, contour_values, contour_values, contour_values])

    # Update elements
    elements_original = elements[:, 1:4] - 1  # Convert to zero-based index
    elements_mirror_x = elements_original + num_original_nodes
    elements_mirror_y = elements_mirror_x + num_original_nodes
    elements_mirror_xy = elements_mirror_y + num_original_nodes

    # Combine all elements
    combined_elements = np.vstack([elements_original, elements_mirror_x, elements_mirror_y, elements_mirror_xy])
    return combined_nodes, combined_elements, combined_contour_values
def contourPlotter(lx,ly,lz,nd_defx,nd_defy,nd_defz,nd_fail,dispVal,ForceVal): 
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'xy'}, {'type': 'scene'}]],  # Specify types of plots
        subplot_titles=("Force Displacement", "Failure Contour Plot")
    )
    fig.add_trace(
        go.Scatter(x=dispVal, y=ForceVal, mode='lines', name='Force Displacement'),
        row=1, col=1
    )
    filename = 'standardmeshdataRSC.json'
    with open(filename, 'r') as file:
        data = json.load(file)

    # Extract nodal and element data
    node_data_1 = np.array(data['nd_data'])
    element_data_1 = np.array(data['ele_data'])
    nodal_failure_data_1 = nd_fail
    lx_scale = lx
    ly_scale = ly
    lz_scale = lz
    node_data, element_data, nodal_failure_data = mirror_mesh(node_data_1,element_data_1,nodal_failure_data_1)

    # Convert lists to numpy arrays
    nd_defx = np.array(nd_defx)
    nd_defy = np.array(nd_defy)
    nd_defz = np.array(nd_defz)
    nd_fail = np.array(nd_fail)
    dispVal = np.array(dispVal)
    ForceVal = np.array(ForceVal)

    # Check and adjust shapes of nodal deformations
    num_nodes = node_data.shape[0]
    print(node_data.shape)
    if len(nd_defx) != num_nodes:
        raise ValueError(f"Shape mismatch: nd_defx has {len(nd_defx)} elements but expected {num_nodes}.")
    if len(nd_defy) != num_nodes:
        raise ValueError(f"Shape mismatch: nd_defy has {len(nd_defy)} elements but expected {num_nodes}.")
    if len(nd_defz) != num_nodes:
        raise ValueError(f"Shape mismatch: nd_defz has {len(nd_defz)} elements but expected {num_nodes}.")

    # Prepare coordinates and indices for Plotly
    x, y, z = node_data[:, 1]*lx_scale+nd_defx, node_data[:, 2]*ly_scale+nd_defy, node_data[:, 3]*lz_scale+nd_defz
    I, J, K = element_data.T  # Convert to zero-based index

    fig.add_trace(
        go.Mesh3d(
            x=x, y=y, z=z, 
            i=I, j=J, k=K, 
            intensity=nodal_failure_data,  # Use nodal_values for coloring
            colorscale='Viridis',
            name='Mesh',
            showscale=True
        )
    )  
    import plotly.offline as pyo

    # Save to HTML
    pyo.plot(fig, filename='composite_figure.html', auto_open=True)
    print(predictions)