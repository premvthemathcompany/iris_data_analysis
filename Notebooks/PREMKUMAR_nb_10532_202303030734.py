#!/usr/bin/env python
# coding: utf-8

# ## This is your Downloaded Blueprint Notebook ##

# In[31]:


# tags to identify this iteration when submitted
# example: codex_tags = {'env': 'dev', 'region': 'USA', 'product_category': 'A'}

codex_tags = {
    'env': 'dev', 'region': 'USA', 'product_category': 'A'
}

from codex_widget_factory import utils
results_json=[]


# ### IRIS CSV Data

# In[32]:


irisCsvTableData = """
import pandas as pd
import json
from sqlalchemy import create_engine

APPLICATION_DB_HOST = "trainingserverbatch3.postgres.database.azure.com"
APPLICATION_DB_NAME = "Training_S3_DB"
APPLICATION_DB_USER = "Trainingadmin"
APPLICATION_DB_PASSWORD = "p%40ssw0rd"

def getLogger():
    import logging
    logging.basicConfig(filename="UIACLogger.log",
                        format='%(asctime)s %(message)s',
                        filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger


logger = getLogger()


def read_database_data(sql_query, filename):
    logger.info(f"Read dataset file: {filename}")
    connection = None
    try:
        connection_uri = f"postgresql://{APPLICATION_DB_USER}:{APPLICATION_DB_PASSWORD}@{APPLICATION_DB_HOST}/{APPLICATION_DB_NAME}"
        engine = create_engine(connection_uri)
        connection = engine.connect()
        dframe = pd.read_sql_query(sql_query, con=connection)
        return dframe
    except Exception as error_msg:
        print(f"Error occured while reading data from database using query {sql_query} and error info: {error_msg}")
    finally:
        if connection is not None:
            connection.close()
            

def get_filter_table(dframe, selected_filters):
    logger.info("Applying screen filters on the grid table dframe.")
    select_df = dframe.copy()
    for item in list(selected_filters):
        if isinstance(selected_filters[item], list):
            if 'All' not in selected_filters[item] and selected_filters[item]:
                select_df = select_df[select_df[item].isin(
                    selected_filters[item])]
        else:
            if selected_filters[item] != 'All':
                select_df = select_df[select_df[item]
                                      == selected_filters[item]]
    logger.info("Successfully applied screen filters on the grid table dframe.")
    return select_df


def generate_dynamic_table(dframe, name='Sales', grid_options={"tableSize": "small", "tableMaxHeight": "80vh", "quickSearch":True}, group_headers=[], grid="auto"):
    logger.info("Generate dynamic Grid table json from dframe")
    table_dict = {}
    table_props = {}
    table_dict.update({"grid": grid, "type": "tabularForm",
                      "noGutterBottom": True, 'name': name})
    values_dict = dframe.dropna(axis=1).to_dict("records")
    table_dict.update({"value": values_dict})
    col_def_list = []
    for col in list(dframe.columns):
        col_def_dict = {}
        col_def_dict.update({"headerName": col, "field": col})
        col_def_list.append(col_def_dict)
    table_props["groupHeaders"] = group_headers
    table_props["coldef"] = col_def_list
    table_props["gridOptions"] = grid_options
    table_dict.update({"tableprops": table_props})
    logger.info("Successfully generated dynamic Grid table json from dframe")
    return table_dict


def build_grid_table_json():
    logger.info("Preparing grid table json for Historical Screen")
    form_config = {}
    filename = "i01654_irisdata"
    sql_query = f"select * from {filename}"    
    dframe = read_database_data(sql_query, filename)
    # selected_filters = {"target": 'Iris-setosa'} # for testing
    dframe = get_filter_table(dframe, selected_filters)
    form_config['fields'] = [generate_dynamic_table(dframe)]
    grid_table_json = {}
    grid_table_json['form_config'] = form_config
    logger.info("Successfully prepared grid table json for Historical Screen")
    return grid_table_json

grid_table_json = build_grid_table_json()
dynamic_outputs = json.dumps(grid_table_json)

# for testing
# print(dynamic_outputs)

"""


# ### IRIS CSV Table Filter

# In[33]:


irisCsvTableDataFilter = """
import pandas as pd
import json
from itertools import chain
from sqlalchemy import create_engine

APPLICATION_DB_HOST = "trainingserverbatch3.postgres.database.azure.com"
APPLICATION_DB_NAME = "Training_S3_DB"
APPLICATION_DB_USER = "Trainingadmin"
APPLICATION_DB_PASSWORD = "p%40ssw0rd"

def getLogger():
    import logging
    logging.basicConfig(filename="UIACLogger.log",
                        format='%(asctime)s %(message)s',
                        filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger


logger = getLogger()


def read_database_data(sql_query, filename):
    logger.info(f"Read dataset file: {filename}")
    connection = None
    try:
        connection_uri = f"postgresql://{APPLICATION_DB_USER}:{APPLICATION_DB_PASSWORD}@{APPLICATION_DB_HOST}/{APPLICATION_DB_NAME}"
        engine = create_engine(connection_uri)
        connection = engine.connect()
        dframe = pd.read_sql_query(sql_query, con=connection)
        return dframe
    except Exception as error_msg:
        print(f"Error occured while reading data from database using query {query} and error info: {error_msg}")
    finally:
        if connection is not None:
            connection.close()


def get_response_filters(current_filter_params, df, default_values_selected, all_filters, multi_select_filters, extra_filters={}):
    logger.info("Preparing filter dictionary")
    filters = list(df.columns)
    default_values_possible = {}
    for item in filters:
        default_possible = list(df[item].unique())
        if item in all_filters:
            default_possible = list(chain(['All'], default_possible))
        default_values_possible[item] = default_possible
    if extra_filters:
        filters.extend(list(extra_filters.keys()))
        default_values_possible.update(extra_filters)
    if current_filter_params:
        selected_filters = current_filter_params["selected"]
        # print(selected_filters)
        # current_filter = current_filter_params[selected_filters]
        # current_index = filters.index(current_filter)
        select_df = df.copy()
    final_dict = {}
    iter_value = 0
    data_values = []
    default_values = {}
    for item in filters:
        filter_dict = {}
        filter_dict["widget_filter_index"] = int(iter_value)
        filter_dict["widget_filter_function"] = False
        filter_dict["widget_filter_function_parameter"] = False
        filter_dict["widget_filter_hierarchy_key"] = False
        filter_dict["widget_filter_isall"] = True if item in all_filters else False
        filter_dict["widget_filter_multiselect"] = True if item in multi_select_filters else False
        filter_dict["widget_tag_key"] = str(item)
        filter_dict["widget_tag_label"] = str(item)
        filter_dict["widget_tag_input_type"] = "select",
        filter_dict["widget_filter_dynamic"] = True
        if current_filter_params:
            if item in df.columns:
                possible_values = list(select_df[item].unique())
                item_default_value = selected_filters[item]
                if item in all_filters:
                    possible_values = list(chain(['All'], possible_values))
                if item in multi_select_filters:
                    for value in selected_filters[item]:
                        if value not in possible_values:
                            if possible_values[0] == "All":
                                item_default_value = possible_values
                            else:
                                item_default_value = [possible_values[0]]
                else:
                    if selected_filters[item] not in possible_values:
                        item_default_value = possible_values[0]
                filter_dict["widget_tag_value"] = possible_values
                if item in multi_select_filters:
                    if 'All' not in item_default_value and selected_filters[item]:
                        select_df = select_df[select_df[item].isin(
                            item_default_value)]
                else:
                    if selected_filters[item] != 'All':
                        select_df = select_df[select_df[item]
                                              == item_default_value]
            else:
                filter_dict["widget_tag_value"] = extra_filters[item]
        else:
            filter_dict["widget_tag_value"] = default_values_possible[item]
            item_default_value = default_values_selected[item]
        data_values.append(filter_dict)
        default_values[item] = item_default_value
        iter_value = iter_value + 1
    final_dict["dataValues"] = data_values
    final_dict["defaultValues"] = default_values
    logger.info("Successfully prepared filter dictionary")
    return final_dict


def prepare_filter_json():
    logger.info(f"Preparing json for Filters in Iris Grid Table")
    # Prepare Filter json for Target in the Iris Grid Table.
    filename = "i01654_irisdata"
    sql_query = f"select * from {filename}"    
    dframe = read_database_data(sql_query, filename)
    dframe = dframe.groupby(['target']).sum().reset_index()
    filter_dframe = dframe[['target']]
    default_values_selected = {'target': 'Iris-setosa'}
    all_filters = []
    multi_select_filters = []
    # current_filter_params = {"selected": default_values_selected}
    final_dict_out = get_response_filters(
        current_filter_params, filter_dframe, default_values_selected, all_filters, multi_select_filters)
    logger.info(f"Successful prepared json for Filters in Iris Data Grid Table")
    return json.dumps(final_dict_out)


dynamic_outputs = prepare_filter_json()
"""


# ### Scatterplot

# In[34]:


irisScatterplot = """
import plotly.express as px
import pandas as pd
import json
import plotly.io as io
from sqlalchemy import create_engine

APPLICATION_DB_HOST = "trainingserverbatch3.postgres.database.azure.com"
APPLICATION_DB_NAME = "Training_S3_DB"
APPLICATION_DB_USER = "Trainingadmin"
APPLICATION_DB_PASSWORD = "p%40ssw0rd"


def getLogger():
    import logging
    logging.basicConfig(filename="UIACLogger.log",
                        format='%(asctime)s %(message)s',
                        filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger


logger = getLogger()


def read_database_data(sql_query, filename):
    logger.info(f"Read dataset file: {filename}")
    connection = None
    try:
        connection_uri = f"postgresql://{APPLICATION_DB_USER}:{APPLICATION_DB_PASSWORD}@{APPLICATION_DB_HOST}/{APPLICATION_DB_NAME}"
        engine = create_engine(connection_uri)
        connection = engine.connect()
        dframe = pd.read_sql_query(sql_query, con=connection)
        return dframe
    except Exception as error_msg:
        print(f"Error occured while reading data from database using query {query} and error info: {error_msg}")
    finally:
        if connection is not None:
            connection.close()


def getGraph(dframe, filters):
    logger.info(
        "Preparing bar graph json to understand products sales over time")
    for item in filters:
        if 'All' in filters[item]:
            continue
        elif isinstance(filters[item], list):
            dframe = dframe[dframe[item].isin(filters[item])]
        else:
            dframe = dframe[dframe[item] == filters[item]]
    
    fig = px.scatter(dframe, x="sepalwidthcm", y="sepallengthcm", color="target",
                     size='petallengthcm', hover_data=['petalwidthcm'])
    # fig.show()
    
    logger.info("Successfully prepared bar graph json to understand products sales over time")
    return io.to_json(fig)


# selected_filters = {"target": 'Iris-versicolor'}

filename = "i01654_irisdata"
sql_query = f"select * from {filename}"    
dframe = read_database_data(sql_query, filename)
dynamic_outputs = getGraph(dframe, selected_filters)

"""


# ### Color Table

# In[35]:


color_code_str = '''
from plotly.io import to_json
def get_grid_table(df,
                   col_props={},
                   grid_options={"tableSize": "small", "tableMaxHeight": "80vh"},
                   group_headers=[],
                   popups={},
                   popup_col=[],
                   popup_col_props={},
                   popup_grid_options={},
                   popup_group_headers={}
                   ):
    comp_dict = {}
    comp_dict['is_grid_table'] = True
    comp_props_dict = {}
    actual_columns = df.columns[~ ((df.columns.str.contains("_bgcolor")) | (df.columns.str.contains("_color")))]
    bg_color_columns = df.columns[df.columns.str.contains("_bgcolor")]
    color_columns = df.columns[df.columns.str.contains("_color")]
    values_dict = df[actual_columns].to_dict("records")
    row_props_list = []
    for index, row_values in enumerate(values_dict):
        row_props_dict = {}
        for col_name, row_value in row_values.items():
            row_props_dict[col_name] = row_value
            if (col_name in popup_col) or (col_name + '_bgcolor' in bg_color_columns) or (col_name + '_color' in color_columns):
                row_props_dict[col_name + '_params'] = {}
                if col_name in popup_col:
                    insights_grid_options = popup_grid_options.copy()
                    insights_grid_options.update({"tableTitle": row_value})
                    if isinstance(popups[col_name][row_value], pd.DataFrame):
                        row_props_dict[col_name + '_params'].update({"insights": {
                            "data": get_grid_table(popups[col_name][row_value], col_props=popup_col_props[col_name], grid_options=insights_grid_options[col_name],
                                                   group_headers=popup_group_headers[col_name])
                        }})
                    elif isinstance(popups[col_name][row_value], go.Figure):
                        row_props_dict[col_name + '_params'].update({"insights": {
                            "data": json.loads(to_json(popups[col_name][row_value]))
                        }})
                if col_name + '_bgcolor' in bg_color_columns:
                    row_props_dict[col_name + '_params'].update({'bgColor': df.iloc[index].to_dict()[col_name + '_bgcolor']})
                if col_name + '_color' in color_columns:
                    row_props_dict[col_name + '_params'].update({'color': df.iloc[index].to_dict()[col_name + '_color']})
        row_props_list.append(row_props_dict)
    col_props_list = []
    for col in actual_columns:
        col_props_dict = {}
        col_props_dict.update({"headerName": col, "field": col, 'cellParamsField': col + '_params'})
        if col in popup_col:
            col_props_dict.update({"enableCellInsights": True})
        if col in col_props:
            col_props_dict.update(col_props[col])
        col_props_list.append(col_props_dict)
    comp_props_dict['rowData'] = row_props_list
    comp_props_dict["coldef"] = col_props_list
    comp_props_dict["gridOptions"] = grid_options
    if group_headers:
        comp_props_dict['groupHeaders'] = group_headers
    comp_dict.update({"tableProps": comp_props_dict})
    return comp_dict
'''


# In[36]:


color_table_code_str = '''
from sklearn.datasets import load_iris
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine

APPLICATION_DB_HOST = "trainingserverbatch3.postgres.database.azure.com"
APPLICATION_DB_NAME = "Training_S3_DB"
APPLICATION_DB_USER = "Trainingadmin"
APPLICATION_DB_PASSWORD = "p%40ssw0rd"

def getLogger():
    import logging
    logging.basicConfig(filename="UIACLogger.log",
                        format='%(asctime)s %(message)s',
                        filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger


logger = getLogger()

def read_database_data(sql_query, filename):
    logger.info(f"Read dataset file: {filename}")
    connection = None
    try:
        connection_uri = f"postgresql://{APPLICATION_DB_USER}:{APPLICATION_DB_PASSWORD}@{APPLICATION_DB_HOST}/{APPLICATION_DB_NAME}"
        engine = create_engine(connection_uri)
        connection = engine.connect()
        dframe = pd.read_sql_query(sql_query, con=connection)
        return dframe
    except Exception as error_msg:
        print(f"Error occured while reading data from database using query {query} and error info: {error_msg}")
    finally:
        if connection is not None:
            connection.close()


filename = "i01654_irisdata"
sql_query = f"select * from {filename}"    
main_df = read_database_data(sql_query, filename)
main_df['sepallengthcm_bgcolor'] = main_df['sepallengthcm'].apply(lambda x: "#0000FF" if x>5 else "#FF0000")
main_df['sepalwidthcm_bgcolor'] = '#FF00FF'
main_df['petallengthcm_bgcolor'] = '#00FFFF'
main_df['petalwidthcm_bgcolor'] = '#800000'
main_df['target_bgcolor'] = '#00FF00'



container_dict = {}
container_dict = get_grid_table(main_df)
dynamic_outputs = json.dumps(container_dict)
# print(dynamic_outputs)
'''

irisColorTable = color_code_str + color_table_code_str


# ### Graph

# In[37]:


irisQuadraticGraph = """
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import plotly.io as io


def getLogger():
    import logging
    logging.basicConfig(filename="UIACLogger.log",
                        format='%(asctime)s %(message)s',
                        filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger


logger = getLogger()


def plotGraph():
    x = np.arange(10)
    fig = go.Figure(data=go.Scatter(x=x, y=x**2))
     
    # fig.show()
    
    logger.info("Successfully prepared x=y^2")
    return io.to_json(fig)


# selected_filters = {"target": 'Iris-versicolor'}

dynamic_outputs = plotGraph()

"""


# ### Multi Trace Plots

# In[38]:


irisMultiTracePlots = """
import plotly.graph_objects as go
import pandas as pd
import json
import plotly.io as io
from sqlalchemy import create_engine

APPLICATION_DB_HOST = "trainingserverbatch3.postgres.database.azure.com"
APPLICATION_DB_NAME = "Training_S3_DB"
APPLICATION_DB_USER = "Trainingadmin"
APPLICATION_DB_PASSWORD = "p%40ssw0rd"


def getLogger():
    import logging
    logging.basicConfig(filename="UIACLogger.log",
                        format='%(asctime)s %(message)s',
                        filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger


logger = getLogger()


def read_database_data(sql_query, filename):
    logger.info(f"Read dataset file: {filename}")
    connection = None
    try:
        connection_uri = f"postgresql://{APPLICATION_DB_USER}:{APPLICATION_DB_PASSWORD}@{APPLICATION_DB_HOST}/{APPLICATION_DB_NAME}"
        engine = create_engine(connection_uri)
        connection = engine.connect()
        dframe = pd.read_sql_query(sql_query, con=connection)
        return dframe
    except Exception as error_msg:
        print(f"Error occured while reading data from database using query {query} and error info: {error_msg}")
    finally:
        if connection is not None:
            connection.close()


def getGraph(dframe, filters):
    logger.info("Preparing bar graph json to understand products sales over time")
    for item in filters:
        if 'All' in filters[item]:
            continue
        elif isinstance(filters[item], list):
            dframe = dframe[dframe[item].isin(filters[item])]
        else:
            dframe = dframe[dframe[item] == filters[item]]
    
    
    fig = go.Figure(data=[go.Scatter(
     x=dframe['sepalwidthcm'],
     y=dframe['sepallengthcm'],
     mode='markers',
     marker=dict(
         color='blue',
     ))])

    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=["y", "petallengthcm"],
                        label="Petal length",
                        method="restyle"
                    ),
                    dict(
                        args=["y", "petalwidthcm"],
                        label="Petal Width",
                        method="restyle"
                    )
                ]),
                direction="down",
            ),
        ]
    )

    # fig.show()
    
    logger.info("Successfully prepared bar graph json to understand products sales over time")
    return io.to_json(fig)


# selected_filters = {"target": 'Iris-versicolor'}

filename = "i01654_irisdata"
sql_query = f"select * from {filename}"    
dframe = read_database_data(sql_query, filename)
dynamic_outputs = getGraph(dframe, selected_filters)

"""


# ### Result JSON

# In[39]:


dynamic_result = {
    'Plot': irisScatterplot,
    'Filter Table': irisCsvTableData,
    'Color Table': irisColorTable,
    'Graph': irisQuadraticGraph,
    'Multitrace': irisMultiTracePlots,
    
}

dynamic_filter = {
    'irisCsvTableDataFilter': irisCsvTableDataFilter,
}

results_json.append({
    'type': 'dynamic',
    'name': 'Iris Analysis',
    'component': 'Analysis component',
    'dynamic_visual_results': dynamic_result,
    'dynamic_code_filters': dynamic_filter,
})


# ### Please save and checkpoint notebook before submitting params

# In[40]:



currentNotebook = 'PREMKUMAR_nb_10532_202303030734.ipynb'

get_ipython().system('jupyter nbconvert --to script {currentNotebook} ')


# In[ ]:



utils.submit_config_params(url='https://codex-api-stage.azurewebsites.net/codex-api/projects/upload-config-params/KlHkQqeyVCHgjeU5Z61yxQ', nb_name=currentNotebook, results=results_json, codex_tags=codex_tags, args={})


# In[ ]:




