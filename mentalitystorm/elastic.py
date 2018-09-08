from elasticsearch import Elasticsearch, ElasticsearchException
from .observe import View
import pprint
import logging

log = logging.getLogger(__name__)


def connect(host, port):
    if host is None:
        import socket
        host = socket.gethostname()
    return Elasticsearch([{'host': host, 'port': port}])


class ElasticSearchUpdater(View):
    def __init__(self, host=None, port=9200):
        self. es = connect(host, port)

    def update(self, data, metadata):
        try:
            res = self.es.index(index="models", doc_type='model', id=metadata['filename'], body=metadata)
        except ElasticsearchException as es1:
            log.error("ElasticSearch threw exception")
            log.error(es1)

    def register(self, model):
        model.registerView('save', self)


class ElasticSetup:
    def __init__(self, host=None, port=9200):
        self.es = connect(host, port)

    def createModelIndex(self):
        body = {
            "settings" : {
                "number_of_shards" : 1
            },
            "mappings" : {
                "model" : {
                    "properties" : {
                        "guid" : { "type" : "keyword" }
                    }
                }
            }
        }
        self.es.indices.create(index='models', body=body)

    def deleteModelIndex(self):
        try:
            self.es.indices.delete(index='models')
        except ElasticsearchException as es1:
            log.error("ElasticSearch threw exception")
            log.error(es1)

# elastic seems a bit too complex, think I'll just use python
class ElasticQueryTool:
    def __init__(self, host=None, port=9200):
        self. es = connect(host, port)

    def queryAll(self):

        body = { "from": 0, "size": 30,
                 "sort" : [],
            "query": {
                "match_all": {}
            }
        }
        res = self.es.search(index="models", body=body)

        print("Got %d Hits:" % res['hits']['total'])
        for hit in res['hits']['hits']:
            print("%(timestamp)s %(guid)s: %(ave_test_loss)s" % hit["_source"])

    def mostImproved(self):

        body =  {
            "from": 0, "size": 30,
            "aggs" : {
                    "model_by_unique_guid" : {
                        "terms" : { "field" : "guid" }
                    }
                }
        }

        res = self.es.search(index="models", body=body)
        print("Got %d Hits:" % res['hits']['total'])
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(res)

        for hit in res['hits']['hits']:
            print("%(timestamp)s %(guid)s: %(ave_test_loss)s" % hit["_source"])

    # def top_hits(self):
    #     body = {
    #         "from": 0, "size": 30,
    #         "aggs": {
    #             "model_by_unique_guid": {
    #                 "terms": {
    #                     "field": "guid"
    #                 },
    #                 "aggs": {
    #                     "top_thingo_hits": {
    #                         "top_hits": {
    #                             "sort": [
    #                                 {
    #                                     "date": {
    #                                         "order": "desc"
    #                                     }
    #                                 }
    #                             ],
    #                             "_source": {
    #                                 "includes": [ "date", "price" ]
    #                             },
    #                             "size" : 1
    #                         }
    #                     }
    #                 }
    #             }
    #         }
    #     }

