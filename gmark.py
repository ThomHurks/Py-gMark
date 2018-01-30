#! /usr/bin/env python3

import sys
import argparse
import numpy
from random import shuffle
from lxml import etree
import rstr
from json import dumps
from copy import deepcopy

__author__ = 'Thom Hurks'


def parse_args():
    parser = argparse.ArgumentParser(prog='gMark', description='Given a graph configuration, generate a new graph.')
    parser.add_argument('schema', metavar='schema_file', type=open, help='The graph schema XML file')
    parser.add_argument('--showschema', action='store_true', help='Show the parsed graph schema')
    parser.add_argument('--silentrun', action='store_true', help='Don\'t generate a graph, but simulate silently')
    return parser.parse_args()


def generate_edges(graph_configuration, silent_run):
    schema = graph_configuration['schema']
    constraints = schema['constraints']
    distributions = schema['distributions']
    for distribution in distributions:
        v_src = []
        v_trg = []
        nr_source_nodes = constraints[distribution['source']]
        draws = draw_distribution(distribution['out_distribution'], nr_source_nodes)
        if draws is not None:
            for i in range(nr_source_nodes):
                times = max(int(draws[i]), 0)
                v_src.extend([i for _ in range(times)])
        nr_target_nodes = constraints[distribution['target']]
        draws = draw_distribution(distribution['in_distribution'], nr_target_nodes)
        if draws is not None:
            for j in range(nr_target_nodes):
                times = max(int(draws[j]), 0)
                v_trg.extend([j for _ in range(times)])
        shuffle(v_src)
        shuffle(v_trg)
        nr_edges = min(len(v_src), len(v_trg))
        allow_loops = distribution['allow_loops']
        if not silent_run:
            for i in range(nr_edges):
                source = v_src[i]
                target = v_trg[i]
                if allow_loops or source != target:
                    print('{},{},{},{}'.format(distribution['source'], source, distribution['predicate'], target))


def generate_nodes(graph_configuration, silent_run):
    schema = graph_configuration['schema']
    constraints = schema['constraints']
    types = schema['types']
    for (node_type, attributes) in types.items():
        nr_nodes = constraints[node_type]
        for attribute in attributes:
            name = attribute['name']
            kind = attribute['type']
            prefix = '{},{},'.format(node_type, name)
            if kind == 'regex':
                for i in range(nr_nodes):
                    value = rstr.xeger(attribute['regex'])
                    if not silent_run:
                        print('{}{},{}'.format(prefix, i, value))
            elif kind == 'categorical':
                randoms = draw_distribution({'name': 'random'}, nr_nodes)
                categories = []
                cumulative_probabilities = []
                cumulative_probability = 0
                for (category, probability) in attribute['categories'].items():
                    cumulative_probability += probability
                    cumulative_probabilities.append(cumulative_probability)
                    categories.append(category)
                category_range = range(len(categories))
                for i in range(nr_nodes):
                    random = randoms[i]
                    for cat_index in category_range:
                        if random < cumulative_probabilities[cat_index] and not silent_run:
                            print('{}{},{}'.format(prefix, i, categories[cat_index]))
                            break
            elif kind == 'numeric':
                numbers = draw_distribution(attribute['distribution'], nr_nodes)
                attr_min = attribute['min']
                attr_max = attribute['max']
                if attr_min or attr_max:
                    numpy.clip(numbers, attr_min, attr_max, out=numbers)
                # TODO: make nr of decimals configurable.
                numpy.around(numbers, decimals=0, out=numbers)
                # TODO: only cast to int when nr of decimals = 0 (to remove the decimal .0)
                numbers = numbers.astype(int, copy=False)
                for i in range(nr_nodes):
                    if not silent_run:
                        print('{}{},{}'.format(prefix, i, numbers[i]))


def draw_distribution(distribution, number):
    name = distribution['name']
    if name == 'uniform':
        return numpy.random.uniform(low=distribution['min'], high=distribution['max'], size=number)
    elif name == 'gaussian':
        return numpy.random.normal(loc=distribution['mean'], scale=distribution['stdev'], size=number)
    elif name == 'zipfian':
        return numpy.random.zipf(distribution['alpha'], size=number)
    elif name == 'exponential':
        return numpy.random.exponential(scale=distribution['scale'], size=number)
    elif name == 'random':
        return numpy.random.random(size=number)
    else:
        sys.exit('Cannot draw from unknown distribution "{}"'.format(name))


def parse_input_schema(filename):
    relaxng = etree.RelaxNG(etree.parse('schema.rng'))
    parser = etree.XMLParser(remove_blank_text=True)
    document = etree.parse(filename, parser)
    try:
        relaxng.assertValid(document)
    except etree.DocumentInvalid as err:
        sys.exit('Input graph schema invalid: {}'.format(err))
    root = document.getroot()
    size = int(root.get('size'))
    type_nodes = root.find('types').findall('type')
    predicate_nodes = root.find('predicates').findall('predicate')
    (type_names, predicate_names) = get_unique_names(type_nodes, predicate_nodes)
    constraints = get_constraints(type_nodes, size)
    distributions = get_distributions(type_nodes, type_names, predicate_names)
    types = get_types(type_nodes)
    graph_schema = {
        'predicates': predicate_names,
        'types': types,
        'constraints': constraints,
        'distributions': distributions
    }
    return {
        'size': size,
        'schema': graph_schema
    }


def get_distributions(type_nodes, type_names, predicate_names):
    distributions = []
    for typeNode in type_nodes:
        source = typeNode.get('name')
        relations = typeNode.find('relations')
        if relations is None:
            continue
        relations = relations.findall('relation')
        for relation in relations:
            predicate = relation.get('predicate')
            if predicate not in predicate_names:
                sys.exit('Found relation with unspecified predicate "{}"'.format(predicate))
            target = relation.get('target')
            if target not in type_names:
                sys.exit('Found relation with unspecified target type "{}"'.format(target))
            if source != target:
                allow_loops = True
            else:
                allow_loops = relation.get('allow_loops')
                if not allow_loops:
                    allow_loops = False
            affinities = get_affinities(relation.find('affinities'))
            if affinities is not None and source != target:
                sys.exit('Affinities can only be specified on relations between the same node types')
            # TODO: check for duplicate distributions (target+predicate must be unique for this source)
            distributions.append({
                'source': source,
                'target': target,
                'predicate': predicate,
                'allow_loops': allow_loops,
                'in_distribution': parse_distribution(relation.find('inDistribution')),
                'out_distribution': parse_distribution(relation.find('outDistribution')),
                'affinities': affinities
            })
    return distributions


def parse_distribution(distribution_node):
    distribution = distribution_node.find('uniformDistribution')
    if distribution is not None:
        low = float(distribution.get('min'))
        high = float(distribution.get('max'))
        if low > high:
            sys.exit('Invalid uniform distribution found')
        return {
            'name': 'uniform',
            'min': low,
            'max': high
        }
    distribution = distribution_node.find('gaussianDistribution')
    if distribution is not None:
        return {
            'name': 'gaussian',
            'mean': float(distribution.get('mean')),
            'stdev': float(distribution.get('stdev'))
        }
    distribution = distribution_node.find('zipfianDistribution')
    if distribution is not None:
        return {
            'name': 'zipfian',
            'alpha': float(distribution.get('alpha'))
        }
    distribution = distribution_node.find('exponentialDistribution')
    if distribution is not None:
        return {
            'name': 'exponential',
            'scale': float(distribution.get('scale'))
        }
    else:
        sys.exit('Could not parse distribution node "{}"'.format(distribution))


def parse_categories(category_nodes):
    categories = dict()
    total_probability = 0
    uniform_probability = 1 / len(category_nodes)
    for category_node in category_nodes:
        name = category_node.text
        probability = category_node.get('probability')
        if probability:
            probability = float(probability)
        elif total_probability > 0:
            sys.exit('Probability needs to be specified on all categories or none')
        else:
            probability = uniform_probability
        categories[name] = probability
        total_probability += probability
    if abs(total_probability - 1) > (2000 * sys.float_info.epsilon):  # 2000 is just an empirical value.
        sys.exit('The probabilities of the categories need to sum to 1, not "{}"'.format(total_probability))
    return categories


def get_types(type_nodes):
    types = dict()
    for type_node in type_nodes:
        name = type_node.get('name')
        types[name] = get_attributes(type_node.find('attributes'))
    return types


def get_attributes(attribute_nodes):
    attributes = []
    if attribute_nodes is None:
        return attributes
    for attribute_node in attribute_nodes:
        name = attribute_node.get('name')
        required = attribute_node.get('required') == 'true'
        unique = attribute_node.get('unique') == 'true'
        kind = attribute_node.find('numeric')
        if kind is not None:
            low = kind.get('min')
            if low:
                low = float(low)
            high = kind.get('max')
            if high:
                high = float(high)
            if low and high and low > high:
                sys.exit('Invalid min and max attributes for numeric attribute')
            attributes.append({
                'name': name,
                'type': 'numeric',
                'required': required,
                'unique': unique,
                'min': low,
                'max': high,
                'distribution': parse_distribution(kind)
            })
            continue
        kind = attribute_node.find('categorical')
        if kind is not None:
            attributes.append({
                'name': name,
                'type': 'categorical',
                'required': required,
                'unique': unique,
                'categories': parse_categories(kind.findall('category'))
            })
            continue
        kind = attribute_node.find('regex')
        attributes.append({
            'name': name,
            'type': 'regex',
            'required': required,
            'unique': unique,
            'regex': kind.text
        })
    return attributes


def get_affinities(affinity_node):
    type_affinities = dict()
    if affinity_node is not None:
        affinities = affinity_node.findall('attributeAffinity')
        for attribute_affinity in affinities:
            type_affinities[attribute_affinity.get('name')] = {
                'inverse': attribute_affinity.get('inverse'),
                'weight': attribute_affinity.get('weight')
            }
    return type_affinities


def get_unique_names(types, predicates):
    type_names = set()
    predicate_names = set()
    for typeNode in types:
        type_name = typeNode.get('name')
        if type_name in type_names:
            sys.exit('Duplicate entry for type "{}"'.format(type_name))
        type_names.add(type_name)
    for predicate_node in predicates:
        predicate_name = predicate_node.get('name')
        if predicate_name in predicate_names:
            sys.exit('Duplicate entry for predicate "{}"'.format(predicate_name))
        predicate_names.add(predicate_name)
    if not type_names.isdisjoint(predicate_names):
        sys.exit('The type and predicate names overlap')
    return type_names, predicate_names


def get_constraints(types, size):
    constraints = dict()
    for typeNode in types:
        name = typeNode.get('name').strip()
        if name in constraints:
            sys.exit('Duplicate constraint found for type "{}"'.format(name))
        count = typeNode.find('count')
        fixed = count.find('fixed')
        if fixed:
            fixed = int(fixed.text.strip())
        else:
            proportion = float(count.find('proportion').text.strip())
            fixed = int(proportion * size)
        constraints[name] = fixed
    return constraints


def main():
    args = parse_args()
    graph_configuration = parse_input_schema(args.schema)
    if args.showschema:
        print_config = deepcopy(graph_configuration)
        print_config['schema']['predicates'] = list(print_config['schema']['predicates'])
        print(dumps(print_config, indent=4))
    generate_edges(graph_configuration, args.silentrun)
    generate_nodes(graph_configuration, args.silentrun)
    args.schema.close()


if __name__ == "__main__":
    main()
