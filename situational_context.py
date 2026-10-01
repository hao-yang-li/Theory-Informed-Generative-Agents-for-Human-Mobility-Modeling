import copy
import json
import math
import re
from pathlib import Path

from agent_initialization.demographics import education_level, income_level


def configure_context(config):
    source = config.get('situational_context', {}).get('file')
    if not source:
        return config
    config = copy.deepcopy(config)
    context = json.loads(Path(source).read_text())
    prefix = context.get('cbg_prefix') or ''
    if not isinstance(prefix, str) or (prefix and not re.fullmatch(r'[0-9]{1,12}', prefix)):
        raise ValueError('cbg_prefix must be a digit string, empty, or null.')
    name = context.get('name', '')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
        raise ValueError('Context name must contain letters, numbers, underscores or hyphens.')
    if not isinstance(context.get('text'), str) or not context['text'].strip():
        raise ValueError('Context text is required.')
    if len(context.get('category_hints', [])) != 6:
        raise ValueError('Context requires six category hints; use empty strings where appropriate.')
    for task in ('interest', 'preference', 'dynamics'):
        if not isinstance(context.get('openings', {}).get(task), str):
            raise ValueError(f'Context opening is required for {task}.')
    context['templates'] = {
        task: (Path(source).parent / f'{task}.txt').read_text()
        for task in ('interest', 'preference', 'dynamics')
    }
    paths = config['paths']
    prior = Path(paths['policy_functions'])
    paths['policy_functions'] = str(prior.parent / 'contexts' / name / prior.name)
    paths['output_dir'] = str(Path(paths['output_dir']) / config['simulation']['city_name'] / 'contexts' / name)
    if config.get('evaluation', {}).get('hotspot_output'):
        paths_out = Path(config['evaluation']['hotspot_output'])
        config['evaluation']['hotspot_output'] = str(Path(paths['output_dir']) / paths_out.name)
    config['_context'] = context
    return config


def context_prompts(profile, home_poi_probs, categories, context):
    values = dict(agent_sex=profile['sex'], agent_age_group=profile['age_group'],
                  agent_race=profile['race'], agent_industry=profile['industry'],
                  home_cbg_edu_level=education_level(profile),
                  home_cbg_income_level=income_level(profile),
                  home_cbg_poi_probs=home_poi_probs, COMMON_SCENARIO_TEXT=context['text'])
    for i, category in enumerate(categories):
        values[f'category_{i}'] = category
        hint = context['category_hints'][i]
        values[f'hint_{i}'] = f'({hint})' if hint else ''
    return tuple(context['templates'][task].format(
        **values, opening=context['openings'][task])
        for task in ('interest', 'preference', 'dynamics'))


def alpha_multiplier(result):
    value = result.get('alpha_multiplier')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Context policy requires a finite positive alpha_multiplier.')
    return value


def context_cbg_prefix(config):
    return config.get('_context', {}).get('cbg_prefix') or ''


def filter_context_profiles(profiles, field, config):
    prefix = context_cbg_prefix(config)
    return [p for p in profiles if str(p[field]).startswith(prefix)] if prefix else profiles
