from lxml import etree
def merge_svg():
    # Algorithm names and corresponding SVG paths
    algorithm_names = [
        "PSO",
        'Differential Evolution',
        'Dual Annealing',
        'MLSL',
        'CMA-ES',
        'Nelder-Mead',
    ]
    svg_paths = [
        f'./imgs/agg-{name}.svg' for name in algorithm_names
    ]

    # Function to read and parse an SVG file (read as bytes)
    def parse_svg(svg_path):
        with open(svg_path, 'rb') as svg_file:  # Open in binary mode
            svg_content = svg_file.read()
        return etree.fromstring(svg_content)

    # Function to convert width/height values to pixels
    def convert_to_pixels(value):
        if 'px' in value:
            return float(value.replace('px', ''))
        elif 'pt' in value:
            # Convert points to pixels (1pt = 1.333px)
            return float(value.replace('pt', '')) * 1.333
        else:
            # If no recognized unit, just try converting directly to float
            return float(value)

    # Parse each SVG file into an XML tree
    svg_trees = [parse_svg(svg_path) for svg_path in svg_paths]

    # Assuming all SVGs have the same size
    # Extract width and height from the first SVG file
    svg_root = svg_trees[0]
    width = convert_to_pixels(svg_root.attrib['width'])
    height = convert_to_pixels(svg_root.attrib['height'])

    # Define padding size (in pixels)
    padding = -170

    # Calculate the size of the final combined SVG including padding
    combined_width = 2 * width + padding
    combined_height = 3 * height + 2 * padding

    # Create the root element for the combined SVG
    combined_svg = etree.Element('svg', xmlns="http://www.w3.org/2000/svg",
                                width=f'{combined_width}px', height=f'{combined_height}px')

    # Add each SVG into the combined SVG with proper translations for positioning
    for i, svg_tree in enumerate(svg_trees):
        # Create a group element (g) for each SVG
        g_element = etree.Element('g')
        
        # Calculate x and y offsets based on the position in the grid
        x_offset = (i % 2) * (width + padding)
        y_offset = (i // 2) * (height + padding)
        
        # Set the translation for this SVG
        g_element.attrib['transform'] = f'translate({x_offset},{y_offset})'
        
        # Append the content of the parsed SVG to the group element
        for element in svg_tree:
            g_element.append(element)
        
        # Append the group to the combined SVG
        combined_svg.append(g_element)

    # Save the combined SVG to a file
    with open('./imgs/Fig2.svg', 'wb') as svg_file:
        svg_file.write(etree.tostring(combined_svg, pretty_print=True))

    print("Merged SVG has been saved successfully.")
