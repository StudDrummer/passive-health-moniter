/**
 * VIGIL MLInsightsScreen.tsx
 * ===========================
 * Displays ML-scored condition risk from the server's /ml/scores/latest endpoint.
 * Drop alongside App.tsx.
 *
 * Usage in App.tsx:
 *   import MLInsightsScreen from './MLInsightsScreen';
 *   // Add 'insights' to TabName type
 *   // Add tab icon and case in render
 */

import React, {useCallback, useEffect, useState} from 'react';
import {
  ActivityIndicator,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import axios from 'axios';

// ─── Types ────────────────────────────────────────────────────────────────────

interface MLScore {
  condition_id:     string;
  condition_label:  string;
  category:         string;
  probability:      number;
  score_0_100:      number;
  level:            'low' | 'moderate' | 'elevated' | 'high';
  urgent:           boolean;
  insufficient_data:boolean;
  completeness:     number;
  data_quality:     'good' | 'fair' | 'limited';
  published_auc:    number;
  top_signals:      string[];
}

interface MLResult {
  scored_at:    string;
  health_index: number;
  scores:       MLScore[];
}

// ─── Colors (matching App.tsx palette) ────────────────────────────────────────

const C = {
  bg:        '#000000',
  surface2:  '#1C1C1E',
  surface3:  '#2C2C2E',
  border:    '#2C2C2E',
  text:      '#FFFFFF',
  textSub:   '#8E8E93',
  textMuted: '#48484A',
  ok:        '#30D158',
  warn:      '#FF9F0A',
  danger:    '#FF375F',
};

const CAT_COLORS: Record<string, string> = {
  cardiovascular:  '#FF375F',
  neurological:    '#BF5AF2',
  respiratory:     '#64D2FF',
  musculoskeletal: '#1EE8E4',
  metabolic:       '#FF9F0A',
  psychiatric:     '#30D158',
};

const LEVEL_COLOR = (level: string): string => {
  if (level === 'high')     return '#FF375F';
  if (level === 'elevated') return '#FF9F0A';
  if (level === 'moderate') return '#FFD60A';
  return '#30D158';
};

// ─── Score Card ───────────────────────────────────────────────────────────────

function MLScoreCard({score}: {score: MLScore}) {
  const [expanded, setExpanded] = React.useState(false);
  const catColor  = CAT_COLORS[score.category] ?? '#8E8E93';
  const levelColor = LEVEL_COLOR(score.level);
  const lowData   = score.insufficient_data || score.completeness < 0.35;

  return (
    <TouchableOpacity
      onPress={() => setExpanded(p => !p)}
      activeOpacity={0.8}
      style={[s.card, {borderLeftColor: catColor + (lowData ? '50' : 'FF')}]}>

      {/* Header row */}
      <View style={{flexDirection:'row', justifyContent:'space-between', alignItems:'flex-start'}}>
        <View style={{flex:1, marginRight:12}}>
          <Text style={{color: lowData ? C.textMuted : C.text, fontSize:14, fontWeight:'700'}}>
            {score.condition_label}
          </Text>
          <Text style={{color: catColor, fontSize:10, fontWeight:'600',
            textTransform:'uppercase', letterSpacing:0.8, marginTop:2}}>
            {score.category}
          </Text>
        </View>
        <View style={{alignItems:'flex-end'}}>
          {score.insufficient_data ? (
            <Text style={{color: C.textMuted, fontSize:22, fontWeight:'800'}}>—</Text>
          ) : (
            <Text style={{color: levelColor, fontSize:22, fontWeight:'800'}}>
              {score.score_0_100.toFixed(0)}
            </Text>
          )}
          <Text style={{color: levelColor, fontSize:10, fontWeight:'600', textTransform:'uppercase'}}>
            {score.insufficient_data ? 'no data' : score.level}
          </Text>
        </View>
      </View>

      {/* Score bar */}
      {!score.insufficient_data && (
        <View style={{height:3, backgroundColor:C.surface3, borderRadius:2, marginTop:10, marginBottom:6}}>
          <View style={{height:3, width:`${Math.min(100, score.score_0_100)}%`,
            backgroundColor:levelColor, borderRadius:2}}/>
        </View>
      )}

      {/* Data completeness dots */}
      <View style={{flexDirection:'row', gap:3, marginTop: score.insufficient_data ? 10 : 2, marginBottom:4}}>
        {Array.from({length:5}).map((_,i) => (
          <View key={i} style={{width:6, height:6, borderRadius:3,
            backgroundColor: i/5 < score.completeness ? catColor : C.surface3}}/>
        ))}
        <Text style={{color:C.textMuted, fontSize:9, marginLeft:4}}>
          {Math.round(score.completeness*100)}% data · AUC {score.published_auc.toFixed(2)} · {score.data_quality}
        </Text>
      </View>

      {/* Urgent badge */}
      {score.urgent && (
        <View style={{backgroundColor: C.danger+'22', paddingHorizontal:8,
          paddingVertical:3, borderRadius:6, alignSelf:'flex-start', marginBottom:6}}>
          <Text style={{color:C.danger, fontSize:11, fontWeight:'700'}}>
            URGENT — discuss with physician
          </Text>
        </View>
      )}

      {/* Expanded: top signals + disclaimer */}
      {expanded && !score.insufficient_data && (
        <View style={{borderTopWidth:1, borderTopColor:C.border, paddingTop:10, gap:5, marginTop:4}}>
          {score.top_signals.length > 0 && (
            <>
              <Text style={{color:C.textMuted, fontSize:10, fontWeight:'700',
                letterSpacing:1, textTransform:'uppercase', marginBottom:3}}>
                Top Signals
              </Text>
              {score.top_signals.map((sig, i) => (
                <View key={i} style={{flexDirection:'row', gap:8, alignItems:'flex-start'}}>
                  <Text style={{color:catColor, fontSize:11}}>•</Text>
                  <Text style={{flex:1, color:C.textSub, fontSize:12, lineHeight:17}}>{sig}</Text>
                </View>
              ))}
            </>
          )}
          <Text style={{color:C.textMuted, fontSize:10, marginTop:6, lineHeight:14, fontStyle:'italic'}}>
            Score reflects deviation from your personal {score.data_quality === 'good' ? '14+' : '7+'}-day
            baseline. Published model accuracy: AUC {score.published_auc.toFixed(2)}.
            Not a medical diagnosis.
          </Text>
        </View>
      )}

      {score.insufficient_data && (
        <Text style={{color:C.textMuted, fontSize:11, lineHeight:16}}>
          Need {score.data_quality === 'limited' ? '7+' : '14+'} days of synced data
          to compute this score.
        </Text>
      )}

      <Text style={{color:C.textMuted, fontSize:9, marginTop:6}}>
        {expanded ? '▲ Less' : '▼ Signals'}
      </Text>
    </TouchableOpacity>
  );
}

// ─── Category filter ──────────────────────────────────────────────────────────

const CATEGORIES = ['all', 'cardiovascular', 'neurological', 'respiratory',
                    'musculoskeletal'] as const;

// ─── Main Screen ──────────────────────────────────────────────────────────────

export default function MLInsightsScreen({serverUrl}: {serverUrl: string}) {
  const [data,      setData]      = useState<MLResult | null>(null);
  const [loading,   setLoading]   = useState(true);
  const [error,     setError]     = useState<string | null>(null);
  const [refreshing,setRefreshing]= useState(false);
  const [activeCat, setActiveCat] = useState<string>('all');

  const load = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    setError(null);
    try {
      const res = await axios.get(`${serverUrl}/ml/scores/latest`, {timeout: 12000});
      setData(res.data);
    } catch (e: any) {
      setError(e?.message ?? 'Failed to load ML scores');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [serverUrl]);

  useEffect(() => { load(); }, [load]);

  const onRefresh = () => { setRefreshing(true); load(true); };

  // ── Loading ────────────────────────────────────────────────────────────
  if (loading && !data) return (
    <View style={{flex:1, backgroundColor:C.bg, alignItems:'center', justifyContent:'center'}}>
      <ActivityIndicator color={C.ok} size="large"/>
      <Text style={{color:C.textMuted, fontSize:13, marginTop:16}}>
        Loading ML scores…
      </Text>
    </View>
  );

  if (error && !data) return (
    <View style={{flex:1, backgroundColor:C.bg, alignItems:'center',
      justifyContent:'center', paddingHorizontal:32}}>
      <Text style={{color:C.danger, fontSize:16, fontWeight:'700', marginBottom:8}}>
        Cannot reach server
      </Text>
      <Text style={{color:C.textMuted, fontSize:13, textAlign:'center', lineHeight:20}}>
        {error}
      </Text>
      <TouchableOpacity onPress={()=>load()} style={[s.btn, {marginTop:24}]}>
        <Text style={s.btnText}>Retry</Text>
      </TouchableOpacity>
    </View>
  );

  const scores  = data?.scores ?? [];
  const display = activeCat === 'all'
    ? scores
    : scores.filter(sc => sc.category === activeCat);
  const urgentCount = scores.filter(sc => sc.urgent).length;
  const hi     = data?.health_index ?? 100;
  const hiColor = hi >= 70 ? C.ok : hi >= 50 ? C.warn : C.danger;

  return (
    <ScrollView style={{flex:1, backgroundColor:C.bg}}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={C.ok}/>}
      showsVerticalScrollIndicator={false}>

      {/* Health Index tile */}
      <View style={{marginHorizontal:16, marginTop:16, marginBottom:8,
        backgroundColor:C.surface2, borderRadius:16, padding:16,
        flexDirection:'row', alignItems:'center'}}>
        <View style={{flex:1}}>
          <Text style={{color:C.textMuted, fontSize:11, fontWeight:'700',
            letterSpacing:1, textTransform:'uppercase', marginBottom:4}}>
            ML Health Index
          </Text>
          <Text style={{color:hiColor, fontSize:48, fontWeight:'800', letterSpacing:-1}}>
            {scores.length ? hi : '—'}
          </Text>
          <Text style={{color:C.textMuted, fontSize:11, marginTop:2}}>
            {scores.length
              ? `${scores.filter(sc=>!sc.insufficient_data).length}/6 conditions scored`
              : 'Sync 7+ days to compute'}
          </Text>
          {data?.scored_at && (
            <Text style={{color:C.textMuted, fontSize:10, marginTop:4, fontStyle:'italic'}}>
              Scored {new Date(data.scored_at).toLocaleString()}
            </Text>
          )}
        </View>
        {urgentCount > 0 && (
          <View style={{backgroundColor:C.danger+'22', paddingHorizontal:12,
            paddingVertical:8, borderRadius:12, alignItems:'center'}}>
            <Text style={{color:C.danger, fontSize:22, fontWeight:'800'}}>{urgentCount}</Text>
            <Text style={{color:C.danger, fontSize:10, fontWeight:'700'}}>URGENT</Text>
          </View>
        )}
      </View>

      {/* Disclaimer */}
      <View style={{marginHorizontal:16, marginBottom:8, padding:12,
        backgroundColor:'#1a0d00', borderRadius:12, borderWidth:1, borderColor:C.warn+'44'}}>
        <Text style={{color:C.warn, fontSize:11, lineHeight:17}}>
          <Text style={{fontWeight:'700'}}>For research purposes only. </Text>
          These scores reflect statistical patterns in your personal biometric data —
          not clinical diagnoses. Always consult a physician for medical decisions.
          Model accuracy (AUC) is from published studies on population data.
        </Text>
      </View>

      {/* Category filter */}
      <ScrollView horizontal showsHorizontalScrollIndicator={false}
        contentContainerStyle={{paddingHorizontal:16, paddingVertical:8, gap:6, flexDirection:'row'}}>
        {CATEGORIES.map(cat => {
          const isActive = activeCat === cat;
          const color = cat === 'all' ? C.ok : (CAT_COLORS[cat] ?? C.textMuted);
          const count = cat === 'all' ? scores.length
            : scores.filter(sc => sc.category === cat).length;
          const hasAlert = cat !== 'all' &&
            scores.filter(sc => sc.category === cat)
                  .some(sc => sc.level === 'elevated' || sc.level === 'high');
          return (
            <TouchableOpacity key={cat} onPress={() => setActiveCat(cat)}
              style={{paddingHorizontal:14, paddingVertical:7, borderRadius:20,
                backgroundColor: isActive ? color : C.surface2,
                borderWidth:1, borderColor: isActive ? color : C.surface3,
                flexDirection:'row', alignItems:'center', gap:5}}>
              {hasAlert && !isActive && (
                <View style={{width:6, height:6, borderRadius:3, backgroundColor:C.danger}}/>
              )}
              <Text style={{color: isActive ? C.bg : C.textSub,
                fontSize:12, fontWeight:'600'}}>
                {cat.charAt(0).toUpperCase() + cat.slice(1)} ({count})
              </Text>
            </TouchableOpacity>
          );
        })}
      </ScrollView>

      {/* Score cards */}
      {display.length === 0 ? (
        <View style={{alignItems:'center', padding:40}}>
          <Text style={{color:C.textMuted, fontSize:14, textAlign:'center'}}>
            Sync for 7+ days to compute ML risk scores
          </Text>
        </View>
      ) : (
        display.map(sc => <MLScoreCard key={sc.condition_id} score={sc}/>)
      )}

      {/* How it works */}
      <View style={{marginHorizontal:16, marginTop:8, marginBottom:8,
        backgroundColor:C.surface2, borderRadius:14, padding:14}}>
        <Text style={{color:C.text, fontSize:13, fontWeight:'700', marginBottom:8}}>
          How ML scores work
        </Text>
        {[
          ['Personal baseline', 'Scores compare your metrics to your own 14-30 day history, not population averages.'],
          ['Gradient Boosting', '6 calibrated GBM classifiers, one per condition. Trained on clinical literature distributions.'],
          ['Feature importance', 'Each score is driven by specific measurable signals (shown when you tap a card).'],
          ['When to act', 'Elevated (55+) = discuss at next appointment. High (75+) + Urgent = seek medical advice soon.'],
          ['Improve accuracy', 'More days of data = higher completeness dots = more reliable scores. 14+ days is ideal.'],
        ].map(([title, body]) => (
          <View key={title} style={{marginBottom:8}}>
            <Text style={{color:C.ok, fontSize:11, fontWeight:'700'}}>{title}</Text>
            <Text style={{color:C.textSub, fontSize:12, lineHeight:17, marginTop:2}}>{body}</Text>
          </View>
        ))}
      </View>

      <View style={{height:48}}/>
    </ScrollView>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const s = StyleSheet.create({
  card: {
    marginHorizontal:16, marginBottom:8,
    backgroundColor:C.surface2, borderRadius:14,
    padding:14, borderLeftWidth:3,
  },
  btn: {
    padding:16, backgroundColor:C.ok, borderRadius:14,
    alignItems:'center', paddingHorizontal:32,
  },
  btnText: {color:'#000', fontWeight:'700', fontSize:15},
});
