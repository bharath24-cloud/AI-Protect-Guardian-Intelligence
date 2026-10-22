---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: id
    dtype: string
  - name: title
    dtype: string
  - name: url
    dtype: string
  - name: artistUrl
    dtype: string
  - name: artistName
    dtype: string
  - name: artistId
    dtype: string
  - name: completitionYear
    dtype: int64
  - name: dictionaries
    sequence: string
  - name: location
    dtype: string
  - name: period
    struct:
    - name: id
      dtype: string
    - name: title
      dtype: string
  - name: serie
    struct:
    - name: id
      dtype: string
    - name: title
      dtype: string
  - name: genres
    sequence: string
  - name: styles
    sequence: string
  - name: media
    sequence: string
  - name: sizeX
    dtype: float64
  - name: sizeY
    dtype: float64
  - name: diameter
    dtype: float64
  - name: galleries
    sequence: string
  - name: tags
    sequence: string
  - name: description
    dtype: string
  - name: width
    dtype: int64
  - name: height
    dtype: int64
  - name: group
    dtype: string
  splits:
  - name: test
    num_bytes: 4300047463.370933
    num_examples: 63061
  download_size: 4289871770
  dataset_size: 4300047463.370933
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
---



# WikiART Image Datasets

WikiART image dataset is a collection of paintings extracted from [WikiART](https://www.wikiart.org/). 

<p align="center">
<img src="assets/wikiart_general.png" width="650"><br>
</p>


The dataset contains following meta attributes from WikiART entry:
- 'image', 'id', 'title', 'url', 'artistUrl', 'artistName', 'artistId', 'completitionYear', 'dictionaries', 'location', 'period', 'serie', 'genres', 'styles', 'media', 'sizeX', 'sizeY', 'diameter', 'galleries', 'tags', 'description', 'width', 'height', 'group'


***Statistics per Art Movement***

An art movement is assigned to each image.

|  Art Movement | Num of Images |
|:--------------|------:|
| baroque       |  5483 |
| expressionism |  6369 |
| impressionism |  9895 |
| neoclassicism |    14 |
| realism       | 12460 |
| rococo        |  3262 |
| romanticism   | 12993 |
| surrealism    |  4922 |
| symbolism     |  7663 |
| All           | 63061 |

***Statistics per Art Genre***

An image can have multiple genres.


|  Art Genre          |   Num of Images |
|:----------------------|------:|                                                                                                                                            
| abstract              |   809 |                                                                                                                                            
| advertisement         |    15 |                                                                                                                                            
| allegorical painting  |   553 |                                                                                                                                            
| animal painting       |  1111 |                                                                                                                                            
| animation             |     2 |                                                                                                                                            
| architecture          |     3 |
| battle painting       |   364 |
| calligraphy           |     3 |
| capriccio             |    74 |
| caricature            |   374 |
| cityscape             |  3495 |
| cloudscape            |   373 |
| design                |   568 |
| figurative            |  1015 |
| flower painting       |   778 |
| furniture             |     3 |
| genre painting        | 10608 |
| history painting      |   819 |
| icon                  |    15 |
| illustration          |  1372 |
| installation          |    43 |
| interior              |   507 |
| landscape             | 11628 |
| literary painting     |   513 |
| marina                |  1772 |
| miniature             |     1 |
| mosaic                |     2 |
| mythological painting |  1354 |
| nude painting (nu)    |  1449 |
| panorama              |    11 |
| pastorale             |    82 |
| photo                 |   261 |
| portrait              | 13604 |
| poster                |    79 |
| quadratura            |    16 |
| religious painting    |  3087 |
| sculpture             |   657 |
| self-portrait         |   846 |
| sketch and study      |  2452 |
| still life            |  1224 |
| symbolic painting     |  1503 |
| tapestry              |     3 |
| tessellation          |   121 |
| trompe-l'œil          |    19 |
| tronie                |   147 |
| utensil               |     4 |
| vanitas               |    17 |
| veduta                |   165 |
| wildlife painting     |   203 |


