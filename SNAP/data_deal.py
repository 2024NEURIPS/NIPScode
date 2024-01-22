import copy

import pandas as pd

def read_profiles():

    columns = [
        'user_id', 'public', 'completion_percentage', 'gender', 'region',
        'last_login', 'registration', 'AGE', 'body', 'I_am_working_in_field',
        'spoken_languages', 'hobbies', 'I_most_enjoy_good_food', 'pets',
        'body_type', 'my_eyesight', 'eye_color', 'hair_color', 'hair_type',
        'completed_level_of_education', 'favourite_color',
        'relation_to_smoking', 'relation_to_alcohol', 'sign_in_zodiac',
        'on_pokec_i_am_looking_for', 'love_is_for_me', 'relation_to_casual_sex',
        'my_partner_should_be', 'marital_status', 'children',
        'relation_to_children', 'I_like_movies', 'I_like_watching_movie',
        'I_like_music', 'I_mostly_like_listening_to_music',
        'the_idea_of_good_evening', 'I_like_specialties_from_kitchen', 'fun',
        'I_am_going_to_concerts', 'my_active_sports', 'my_passive_sports',
        'profession', 'I_like_books', 'life_style', 'music', 'cars', 'politics',
        'relationships', 'art_culture', 'hobbies_interests',
        'science_technologies', 'computers_internet', 'education', 'sport',
        'movies', 'travelling', 'health', 'companies_brands', 'more'
    ]


    df = pd.read_csv('../data/soc-pokec-profiles.txt', sep='\t', names=columns, index_col=False, header=None, encoding='utf-8')
    df_gender = copy.deepcopy(df)
    df_AGE = copy.deepcopy(df)


    df_gender = df_gender.dropna(subset=['gender'])
    df_gender['gender'] = df_gender['gender'].astype(int)
    selected_gender = df_gender[['user_id', 'gender']]

    selected_gender.to_csv('../data/gender.txt', sep='\t', index=False, header=False)


    df_AGE = df_AGE.dropna(subset=['AGE'])
    df_AGE['AGE'] = df_AGE['AGE'].astype(int)
    selected_AGE = df_AGE[['user_id', 'AGE']]

    selected_AGE.to_csv('../data/AGE.txt', sep='\t', index=False, header=False)
    return df_gender['user_id'], df_AGE['user_id']

def read_relationships(gender_id, AGE_id):
    columns = ['user_id1','user_id2']
    df = pd.read_csv('../data/soc-pokec-relationships.txt', sep='\t', names=columns, index_col=False, header=None, encoding='utf-8')
    gender_df = df[df['user_id1'].isin(gender_id) & df['user_id2'].isin(gender_id)]
    AGE_df = df[df['user_id1'].isin(AGE_id) & df['user_id2'].isin(AGE_id)]
    gender_df.to_csv('../data/gender_relationships.txt', sep='\t', index=False, header=False)
    AGE_df.to_csv('../data/AGE_relationships.txt', sep='\t', index=False, header=False)


def read_gender(k):
    column_profiles = ['user_id','categories']
    column_ralationships = ['user_id1','user_id2']
    data = pd.read_csv('../data/gender.txt', sep='\t', names=column_profiles,
                       index_col=False, header=None, encoding='utf-8')
    relationships = pd.read_csv('../data/gender_relationships.txt', sep='\t', names=column_ralationships,
                                index_col=False, header=None, encoding='utf-8')
    if k != -1:
        data = data.iloc[0:k]
        relationships = relationships[relationships['user_id1'].isin(data['user_id'])
                                      & relationships['user_id2'].isin(data['user_id'])]
    return data, relationships

def read_AGE(k):
    column_profiles = ['user_id','categories']
    column_ralationships = ['user_id1','user_id2']
    data = pd.read_csv('../data/AGE.txt', sep='\t', names=column_profiles,
                       index_col=False, header=None, encoding='utf-8')
    relationships = pd.read_csv('../data/AGE_relationships.txt', sep='\t', names=column_ralationships,
                                index_col=False, header=None, encoding='utf-8')
    if k != -1:
        data = data.iloc[0:k]
        relationships = relationships[relationships['user_id1'].isin(data['user_id'])
                                      & relationships['user_id2'].isin(data['user_id'])]
    return data, relationships



if __name__ == "__main__":
    gender_id , AGE_id = read_profiles()
    read_relationships(gender_id, AGE_id)
