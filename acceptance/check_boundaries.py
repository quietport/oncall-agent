"""Portable boundary regression checks; no credentials, provider calls or reports."""
import sys
import os
sys.dont_write_bytecode=True
from pathlib import Path
import types
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# Never import the user's secret.py, and keep optional vector retrieval offline.
fixture_secret=types.ModuleType('secret')
fixture_secret.API_KEY=''
sys.modules['secret']=fixture_secret
os.environ['DEEPSEEK_API_KEY']=''
os.environ['EMBED_API_KEY']=''
os.environ['DAILY_BUDGET']='1.0'
import app, llm, rag, tools
from agents import chat, triage
from fastapi.testclient import TestClient

class Boundaries(unittest.TestCase):
    def test_vague_alert_blocks_changes_even_with_high_model_confidence(self):
        result=triage.calibrate('【P1】数据库好像有问题，你猜是什么原因',
            [{'output':{'service':'order-service','metrics':{'p99':3}}}],
            {'confidence':0.9,'root_cause':'需要扩容','actions':['立即回滚']})
        self.assertLessEqual(result['confidence'],0.3)
        self.assertTrue(result['needs_information'])
    def test_unknown_tool_fields_are_rejected(self):
        self.assertIn('error',tools.run('get_metrics',{'servce':'pay-service'}))
    def test_uncertainty_removes_speculative_actions(self):
        result=triage.calibrate('order-service 超时',[{'output':{'service':'order-service','metrics':{'p99':3}}}],
            {'confidence':0.2,'root_cause':'证据不足，无法定位','actions':['立即回滚','扩容连接池']})
        self.assertTrue(result['needs_information'])
        self.assertNotIn('回滚',''.join(result['actions']))
        self.assertNotIn('扩容',''.join(result['actions']))
    def test_error_text_cannot_validate_service(self):
        self.assertEqual(triage._evidence_services([{'output':{'error':'pay-service missing'}}]),set())
    def test_successful_service_fields_count(self):
        self.assertEqual(triage._evidence_services([{'output':[{'service':'pay-service','msg':'unknown'}]}]),{'pay-service'})
    def test_payment_steps_retrieved(self):
        hit=rag.get().retrieve('支付服务一直在超时，应该按什么顺序排查？',3)
        self.assertTrue(any(x['doc']=='01-支付超时.md' and x['section']=='排查步骤' for x in hit['items']),hit)
    def test_draft_answer_not_treated_as_delivered(self):
        messages=chat.build_messages('fixture','pay-service 如何？')
        with patch('llm.chat_with_tools',return_value=({'content':'完整答案，错误率18.7%'},0,0)):
            used,_,_=chat._tool_round(messages,llm,[])
        self.assertFalse(used)
        self.assertEqual(messages[-1]['role'],'user')
    def test_budget_applies_to_tool_and_stream_calls(self):
        llm._SPEND[llm._today()]=llm.DAILY_BUDGET
        with self.assertRaises(llm.BudgetExceeded):llm.chat_with_tools([],tools.SPEC)
        with self.assertRaises(llm.BudgetExceeded):list(llm.stream_model([]))
    def test_invalid_conclusion_is_safe_degradation(self):
        with patch('llm.call_model',return_value=('[]',0,0)):
            result,_,_=triage.conclude('pay-service 超时',[],llm)
        self.assertTrue(result['degraded'])
        self.assertNotIn('回滚',''.join(result['actions']))
    def setUp(self):
        llm._CACHE.clear(); llm._SPEND.clear(); chat._memory.clear()
        self.no_http=patch('requests.sessions.Session.request',side_effect=AssertionError('Unexpected external HTTP request'))
        self.no_http.start()
        self.addCleanup(self.no_http.stop)
        self.no_network=patch('llm._post',side_effect=AssertionError('Unexpected model call'))
        self.no_network.start()
        self.client=TestClient(app.app,raise_server_exceptions=False)
    def tearDown(self):
        self.client.close(); self.no_network.stop()
    def test_health_and_six_documents(self):
        self.assertEqual(self.client.get('/health').status_code,200)
        self.assertEqual(len({x['doc'] for x in rag.get().chunks}),6)
    def test_homepage(self):
        self.assertIn('知识库问答',self.client.get('/').text)
    def test_invalid_inputs_rejected(self):
        for path,body in [('/kb/ask',{'question':''}),('/chat/ask',{'question':''}),('/triage',{'alert':'x'})]:
            self.assertEqual(self.client.post(path,json=body).status_code,422)
    def test_bm25_finds_disk_document(self):
        hit=rag.get().retrieve('磁盘写满 日志轮转',3)
        self.assertTrue(any('磁盘' in x['doc'] for x in hit['items']))
    def test_tools_are_read_only_allowlist(self):
        self.assertIn('error',tools.run('delete_server',{}))
        self.assertIn('metrics',tools.run('get_metrics',{'service':'pay-service'}))
    def test_malformed_tool_arguments_do_not_expand_query(self):
        args=chat._parse_args({'function':{'arguments':'{"service":'}})
        out=tools.run('get_metrics',args)
        self.assertIn('error',out,'Malformed JSON becomes {} and queries all services')
    def test_empty_results_not_useful(self):
        self.assertFalse(triage._useful([{'output':[]},{'output':{'error':'missing'}}]))
    def test_query_metadata_is_not_service_evidence(self):
        evidence=[{'tool':'get_metrics','args':{'service':'report-service'},
                   'output':{'error':'没有服务 report-service 的指标'}}]
        result=triage.calibrate('report-service 请求变慢',evidence,{'root_cause':'连接池故障','confidence':0.9})
        self.assertLessEqual(result['confidence'],0.3,'Query/error text incorrectly counts as supporting evidence')
    def test_unrelated_actual_service_lowers_confidence(self):
        result=triage.calibrate('report-service 请求变慢',[{'output':{'service':'pay-service'}}],{'confidence':0.9,'root_cause':'故障'})
        self.assertLessEqual(result['confidence'],0.3)
    def test_replan_attempt_recorded_even_when_still_empty(self):
        p={'hypothesis':'fixture','steps':[{'tool':'missing','args':{}}]}
        with patch('agents.triage.plan',return_value=(p,0,0)),patch('agents.triage.conclude',return_value=({'confidence':0.1,'root_cause':'证据不足'},0,0)):
            result=triage.run('report-service 请求异常',llm)
        self.assertEqual(len(result['plan_rounds']),2,'Second failed planning round disappeared')
    def test_cache_distinguishes_json_format_and_token_budget(self):
        response=unittest.mock.Mock(status_code=200)
        response.json.return_value={'choices':[{'message':{'content':'ordinary text'}}],'usage':{}}
        with patch('llm._post',return_value=response) as call:
            llm.call_model([{'role':'user','content':'same'}],max_tokens=100)
            llm.call_model([{'role':'user','content':'same'}],max_tokens=200,response_format={'type':'json_object'})
        self.assertEqual(call.call_count,2,'Output format and max_tokens are absent from cache key')
    def test_budget_exhaustion_blocks_api_entry(self):
        llm._SPEND[llm._today()]=llm.DAILY_BUDGET
        self.assertEqual(self.client.post('/kb/ask',json={'question':'支付超时'}).status_code,503)
    def test_budget_rechecked_before_each_model_request(self):
        llm._SPEND[llm._today()]=llm.DAILY_BUDGET
        with self.assertRaises(llm.BudgetExceeded):
            llm.call_model([{'role':'user','content':'budget fixture'}])
    def test_separate_session_ids_have_separate_memory(self):
        chat._remember('alpha','user','only alpha')
        self.assertEqual(chat.history('beta'),[])
    def test_memory_retains_six_turns(self):
        for i in range(8):
            chat._remember('alpha','user',str(i));chat._remember('alpha','assistant',str(i))
        self.assertEqual(len(chat.history('alpha')),12)
        self.assertEqual(chat.history('alpha')[0]['content'],'2')
    def test_model_error_returns_502(self):
        with patch('llm.call_model',side_effect=llm.ModelError('fixture unavailable')):
            self.assertEqual(self.client.post('/kb/ask',json={'question':'支付超时'}).status_code,502)
    def test_json_array_plan_rejected_cleanly(self):
        with patch('llm.call_model',return_value=('[]',0,0)):
            self.assertEqual(self.client.post('/triage',json={'alert':'pay-service 请求超时'}).status_code,502,
                             'Valid JSON with wrong structure causes internal error instead of controlled failure')
    def test_stream_plan_schema_error_is_explicit(self):
        with patch('llm.call_model',return_value=('[]',0,0)):
            result=self.client.get('/triage/stream',params={'alert':'pay-service 请求超时'})
        self.assertIn('"stage": "error"',result.text)
        self.assertIn('[DONE]',result.text)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Boundaries)
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    passed=result.testsRun-len(result.failures)-len(result.errors)
    print(f'{passed}/{result.testsRun} passed; synthetic boundary tests, no provider calls.')
    sys.exit(0 if result.wasSuccessful() else 1)
