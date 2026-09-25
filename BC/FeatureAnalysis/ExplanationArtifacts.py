"""Persistence and verification helpers for classifier explanation artifacts."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd


ARTIFACT_VERSION = 1
METADATA_FILENAME = 'explanation.json'


def _value_filename(classifier_name):
    return '{}_shap.csv'.format(classifier_name)


def _feature_filename(classifier_name):
    return '{}_shap_features.csv'.format(classifier_name)


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str,
                      separators=(',', ':'))


def _signature(value):
    return hashlib.sha256(_canonical_json(value).encode('utf-8')).hexdigest()


def _frame_signature(frame):
    return hashlib.sha256(frame.to_csv().encode('utf-8')).hexdigest()


def _validate_aligned_frames(shap_df, feature_df):
    if not isinstance(shap_df, pd.DataFrame) or not isinstance(feature_df, pd.DataFrame):
        raise ValueError('SHAP and feature values must both be DataFrames.')
    if shap_df.empty or feature_df.empty:
        raise ValueError('SHAP and feature values must both be non-empty.')
    if not shap_df.index.equals(feature_df.index):
        raise ValueError('SHAP and feature values must have the same index.')
    if not shap_df.columns.equals(feature_df.columns):
        raise ValueError('SHAP and feature values must have the same columns.')


def _write_text_atomically(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(prefix='.tmp-', dir=str(path.parent), text=True)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='') as handle:
            handle.write(content)
        os.replace(temporary_path, path)
    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def _write_frame_atomically(frame, path):
    _write_text_atomically(path, frame.to_csv())


def _metadata_path(folder):
    return Path(folder) / METADATA_FILENAME


def _write_metadata(folder, metadata):
    _write_text_atomically(_metadata_path(folder),
                           json.dumps(metadata, ensure_ascii=False, indent=2, default=str))
    return metadata


def _remove_classifier_artifacts(folder, classifier_name, include_coefficient=True):
    folder = Path(folder)
    names = [_value_filename(classifier_name), _feature_filename(classifier_name)]
    if include_coefficient:
        names.append('{}_coef.csv'.format(classifier_name))
    for name in names:
        path = folder / name
        if path.is_file():
            path.unlink()


def write_available_explanation(folder, classifier_name, model_params, shap_df,
                                feature_df, explained_data_kind):
    """Write a validated SHAP/value pair and its provenance metadata."""
    _validate_aligned_frames(shap_df, feature_df)
    folder = Path(folder)
    value_filename = _value_filename(classifier_name)
    feature_filename = _feature_filename(classifier_name)
    _write_frame_atomically(shap_df, folder / value_filename)
    _write_frame_atomically(feature_df, folder / feature_filename)
    return _write_metadata(folder, {
        'artifact_version': ARTIFACT_VERSION,
        'classifier_name': classifier_name,
        'status': 'available',
        'reason': '',
        'explained_data_kind': explained_data_kind,
        'model_params_signature': _signature(model_params),
        'shap_filename': value_filename,
        'feature_filename': feature_filename,
        'shap_signature': _frame_signature(shap_df),
        'feature_signature': _frame_signature(feature_df),
    })


def write_unavailable_explanation(folder, classifier_name, model_params, status, reason):
    """Invalidate old explanation files for an unsupported or failed model."""
    if status not in ('unsupported', 'failed'):
        raise ValueError('Unavailable explanation status must be unsupported or failed.')
    _remove_classifier_artifacts(folder, classifier_name)
    return _write_metadata(folder, {
        'artifact_version': ARTIFACT_VERSION,
        'classifier_name': classifier_name,
        'status': status,
        'reason': str(reason),
        'explained_data_kind': None,
        'model_params_signature': _signature(model_params),
        'shap_filename': None,
        'feature_filename': None,
        'shap_signature': None,
        'feature_signature': None,
    })


def _read_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, ValueError, TypeError):
        return None


def load_verified_explanation(folder, classifier_name):
    """Return explanation frames only when their metadata and model match."""
    folder = Path(folder)
    metadata = _read_json(_metadata_path(folder))
    if not isinstance(metadata, dict):
        return None
    if metadata.get('artifact_version') != ARTIFACT_VERSION:
        return None
    if metadata.get('classifier_name') != classifier_name:
        return None
    if metadata.get('status') != 'available':
        return None

    value_filename = _value_filename(classifier_name)
    feature_filename = _feature_filename(classifier_name)
    if metadata.get('shap_filename') != value_filename:
        return None
    if metadata.get('feature_filename') != feature_filename:
        return None

    model_params = _read_json(folder / 'model_param.json')
    if not isinstance(model_params, dict):
        return None
    if metadata.get('model_params_signature') != _signature(model_params):
        return None

    try:
        shap_df = pd.read_csv(folder / value_filename, index_col=0)
        feature_df = pd.read_csv(folder / feature_filename, index_col=0)
        _validate_aligned_frames(shap_df, feature_df)
    except (OSError, ValueError, TypeError, pd.errors.ParserError):
        return None

    if metadata.get('shap_signature') != _frame_signature(shap_df):
        return None
    if metadata.get('feature_signature') != _frame_signature(feature_df):
        return None
    return shap_df, feature_df, metadata
