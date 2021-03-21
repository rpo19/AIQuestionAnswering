def generateTriple(Q, edge):
    # src node
    subj = Q.nodes[edge[0]]['label']
    # edge
    pred = edge[2]['label']
    # dst node
    obj = Q.nodes[edge[1]]['label']
    # generate triple
    triple = ' '.join([subj, pred, obj, '.', '\n'])
    return triple

def generateBody(graph):
    triples = []
    # generate the triple for each edge
    for edge in graph.edges.data():
        triples.append(generateTriple(graph, edge))
    # join triples into the body
    body = ' {\n ' + ' '.join(triples) + ' } \n'
    return body

def getTargetVariable(graph):
    # !!! emergency variable, hope it's not needed !!!
    candidates = []
    # search for variables in nodes
    for node in graph.nodes.data():
        node_id = node[0]
        node_label = node[1]['label']
        # keep only the variable nodes
        if node_label.startswith('?'):
            # !!! emergency operation, hope it's not needed !!!
            candidates.append(node_label)
            # the target variable should be a non-intermediate node
            if graph.out_degree(node_id) == 0 or graph.in_degree(node_id) == 0:
                return node_label
    
    # if we are here I made a wrong assumption lol
    return ' '.join(candidates)

# TODO: put here gabri's code to distinguish between different question types
def generateHeadAndTail(question, variable):
    head = "SELECT " + variable + " WHERE \n"
    tail = " \n "
    return (head, tail)
    

def generateQuery(question, graph):
    # query head and tail
    variable = getTargetVariable(graph)
    (head, tail) = generateHeadAndTail(question, variable)
    # query body
    body = generateBody(graph)
    # assemble query
    query = head + body + tail
    return query
    
   

    