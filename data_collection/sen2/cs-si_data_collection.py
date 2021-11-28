#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
from eodag import EODataAccessGateway
from eodag.plugins.crunch.filter_property import FilterProperty
import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
from eodag import setup_logging
setup_logging(verbose=2)

import matplotlib.image as mpimg


# ### Load Tiles outlines

# In[2]:


# load tile outlines
outline = gpd.read_file("qgis_files/sen2_tiles/tiles_outline.shp")
bretagne = gpd.read_file("qgis_files/admin/bretagne.gpkg")
query_outline = gpd.read_file("qgis_files/admin/query_shape.shp")
query_wkt = query_outline.geometry.to_wkt()[0]

# simplify geometry
#bretagne["geometry"] = bretagne["geometry"].simplify


# In[ ]:





# In[3]:

"""
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
fig, ax = plt.subplots(1, 1,figsize=(3,3), dpi=200)

bretagne.plot(ax=ax,color='lightblue', edgecolor='black')
outline.plot(ax=ax,facecolor="none",edgecolor="red",markersize=0.2)
query_outline.plot(ax=ax,facecolor="none",edgecolor="blue",markersize=0.2)
fig.show()
"""

# ### Query EODAG API

# In[4]:


# API core acess gateway
dag = EODataAccessGateway()


# In[5]:


dag.available_providers("S2_MSI_L2A_MAJA")


# In[6]:


# Query for images
search_results, total_count = dag.search(
    productType='S2_MSI_L2A_MAJA',
    #geom={'lonmin': 1, 'latmin': 43.5, 'lonmax': 2, 'latmax': 44}, # accepts WKT polygons, shapely.geometry, ...
    geom=query_wkt,
    start='2018-04-01',
    end='2018-09-30',
    items_per_page=500
)

# filter images by Cloud Cover
filtered_products = search_results.crunch(FilterProperty({"cloudCover": 5, "operator": "lt"}))

# print info
print("Total No. of images for S2A:", len(search_results))
print("Total No. of images for S2A, max CC 5%:",len(filtered_products))


# ### Check Query

# In[28]:


#for i in filtered_products:
#    print(i)


# In[ ]:





# In[24]:


"""

dag.serialize(filtered_products,filename="tmp/saved_search.json")

deserialized_and_registered = dag.deserialize_and_register("tmp/saved_search.json")
quicklook_path = deserialized_and_registered[0].get_quicklook(base_dir="")
"""


# ### Download Imagery

# In[25]:


# check which ones are online
[p.properties["storageStatus"] for p in filtered_products]


# In[12]:


# filter for online products
#online_search_results = search_results.crunch(FilterProperty(dict(storageStatus="ONLINE")))
#[p.properties["storageStatus"] for p in online_search_results]


# In[ ]:





# In[27]:


# download products
paths = dag.download_all(filtered_products)


# In[ ]:




