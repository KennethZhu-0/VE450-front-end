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

   # Find maximum x and y coordinates
    max_x = np.max(nodes[:, 1])
    max_y = np.max(nodes[:, 2])

    # Mirroring nodes along max_x and max_y
    mirror_x = np.copy(nodes)
    mirror_x[:, 1] = 2 * max_x - mirror_x[:, 1]  # Mirror along max_x

    mirror_y = np.copy(nodes)
    mirror_y[:, 2] = 2 * max_y - mirror_y[:, 2]  # Mirror along max_y

    mirror_xy = np.copy(nodes)
    mirror_xy[:, 1] = 2 * max_x - mirror_xy[:, 1]
    mirror_xy[:, 2] = 2 * max_y - mirror_xy[:, 2]  # Mirror along both max_x and max_y

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
    nd_defx = np.array(nd_defx)
    nd_defy = np.array(nd_defy)
    nd_defz = np.array(nd_defz)
    nd_fail = np.array(nd_fail)
    dispVal = np.array(dispVal)
    ForceVal = np.array(ForceVal)
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
    node_data_1[:, 1] = node_data_1[:, 1]*lx_scale/2+nd_defx
    node_data_1[:, 2] = node_data_1[:, 2]*ly_scale/2+nd_defy
    node_data_1[:, 3] = node_data_1[:, 3]*lz_scale+nd_defz
    node_data, element_data, nodal_failure_data = mirror_mesh(node_data_1,element_data_1,nodal_failure_data_1)
    # #Prepare coordinates and indices for Plotly
    x, y, z = node_data[:, 1], node_data[:, 2], node_data[:, 3]
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
    # print(predictions)