import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add some features to our data.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with data (see load_data)
    Returns
    -------
    with_features : pandas.DataFrame
        DataFrame with some column features added
    """
    df_transformed = df.assign(is_dog = check_is_dog(df['animal_type']),
                               has_name = check_has_name(df['name']),
                               sex = get_sex(df['sex_upon_outcome']),
                               neutered = get_neutered(df['sex_upon_outcome']),
                               hair_type = get_hair_type(df['breed']),
                               days_upon_outcome = compute_days_upon_outcome(df['age_upon_outcome'])
                               )
   
    return df_transformed


def check_is_dog(animal_type: pd.Series) -> pd.Series:
    """Check if the animal is a dog, otherwise return False.
    Parameters
    ----------
    animal_type : pandas.Series
        Type of animal
    Returns
    -------
    result : pandas.Series
        Dog or not
    """
    is_cat_dog = animal_type.str.lower().isin(['dog', 'cat'])
    if not is_cat_dog.all():
        print('Found something else but dogs and cats:\n%s',
              animal_type[~is_cat_dog])
        raise RuntimeError("Found pets that are not dogs or cats.")
    is_dog = animal_type.str.lower() == 'dog'
    return is_dog


def check_has_name(name: pd.Series) -> pd.Series:
    """Check if the animal is not called 'unknown'.
    Parameters
    ----------
    name : pandas.Series
        Animal name
    Returns
    -------
    result : pandas.Series
        has name
    """
    has_name = name.str.lower() != 'unknown'
    return has_name


def get_sex(sex_upon_outcome: pd.Series) -> pd.Series:
    """Determine if the sex was 'Male', 'Female' or unknown.
    Parameters
    ----------
    sex_upon_outcome : pandas.Series
        Sex and fixed state when coming in
    Returns
    -------
    sex : pandas.Series
        Sex when coming in
    """
    sex = pd.Series('unknown', index=sex_upon_outcome.index)
    sex.loc[sex_upon_outcome.str.endswith('Female')] = 'female'
    sex.loc[sex_upon_outcome.str.endswith('Male')] = 'male'
    return sex


def get_neutered(neutered: pd.Series) -> pd.Series:
    """Determine if an animal was intact or not.
    Parameters
    ----------
    neutered : pandas.Series
        Sex and fixed state when coming in
    Returns
    -------
    sex : pandas.Series
        Intact, fixed or unknown
    """
    neutered = neutered.str.lower()
    neutered.loc[neutered.str.contains('neutered')] = 'fixed'
    neutered.loc[neutered.str.contains('spayed')] = 'fixed'

    neutered.loc[neutered.str.contains('intact')] = 'intact'
    neutered.loc[~neutered.isin(['fixed', 'intact'])] = 'unknown'

    return neutered


def get_hair_type(breed: pd.Series) -> pd.Series:
    """Get hair type of a breed.
    Parameters
    ----------
    breed : pandas.Series
        Breed of animal
    Returns
    -------
    hair_type : pandas.Series
        Hair type
    """

    hairType = breed.str.lower()
    Valid_hair_types = ['shorthair', 'medium hair', 'longhair']
    for hair in Valid_hair_types:
        is_hair_type = hairType.str.contains(hair)
        hairType[is_hair_type] = hair

    hairType[~hairType.isin(Valid_hair_types)] = 'unknown'

    return hairType 


def compute_days_upon_outcome(age_upon_outcome: pd.Series) -> pd.Series:
    """Compute age in days upon outcome.
    Parameters
    ----------
    age_upon_outcome : pandas.Series
        Age as string
    Returns
    -------
    days_upon_outcome : pandas.Series
        Age in days
    """
    split_age = age_upon_outcome.str.split()
    time = split_age.apply(lambda x: x[0] if x[0] != 'Unknown' else np.nan)
    period = split_age.apply(lambda x: x[1] if x[0] != 'Unknown' else None)
    period_mapping = {'year': 365, 'years': 365, 'weeks': 7, 'week': 7,
                        'month': 30, 'months': 30, 'days': 1, 'day': 1}
    days_upon_outcome = time.astype(float) * period.map(period_mapping)
    
    return days_upon_outcome

