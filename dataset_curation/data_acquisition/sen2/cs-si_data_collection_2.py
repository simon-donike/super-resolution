#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

# In[3]:


# load tile outlines
outline = gpd.read_file("qgis_files/sen2_tiles/tiles_outline.shp")
bretagne = gpd.read_file("qgis_files/admin/bretagne.gpkg")
query_outline = gpd.read_file("qgis_files/admin/query_shape.shp")
query_wkt = query_outline.geometry.to_wkt()[0]

# simplify geometry
#bretagne["geometry"] = bretagne["geometry"].simplify


# In[ ]:





# In[4]:


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
fig, ax = plt.subplots(1, 1,figsize=(3,3), dpi=200)

bretagne.plot(ax=ax,color='lightblue', edgecolor='black')
outline.plot(ax=ax,facecolor="none",edgecolor="red",markersize=0.2)
query_outline.plot(ax=ax,facecolor="none",edgecolor="blue",markersize=0.2)
fig.show()


# ### Query EODAG API

# In[5]:


# API core acess gateway
dag = EODataAccessGateway()


# In[6]:


dag.available_providers("S2_MSI_L2A_MAJA")


# In[12]:


# Query for images
def query(start,end):
    search_results, total_count = dag.search(
        productType='S2_MSI_L2A_MAJA',
        #geom={'lonmin': 1, 'latmin': 43.5, 'lonmax': 2, 'latmax': 44}, # accepts WKT polygons, shapely.geometry, ...
        geom=query_wkt,
        start=start,
        end=end,
        items_per_page=500
    )

    # filter images by Cloud Cover
    filtered_products = search_results.crunch(FilterProperty({"cloudCover": 5, "operator": "lt"}))

    # print info
    print("Total No. of images for S2A:", len(search_results))
    print("Total No. of images for S2A, max CC 5%:",len(filtered_products))
    paths = dag.download_all(filtered_products)
    
    #return(filtered_products)


# In[19]:


for combi in [("2018-06-30","2018-09-30")]:
    query(combi[0],combi[1])


# ### Check Query

# In[ ]:





# In[ ]:





# In[ ]:





# ### Download Imagery

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




