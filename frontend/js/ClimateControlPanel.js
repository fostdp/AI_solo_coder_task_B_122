/**
 * ClimateControlPanel - 微气候调控控制面板组件
 * 展示当前气候状态、推荐调控动作、投影效果对比
 * 支持手动调节动作参数，实时预览效果
 *
 * 用法：
 *   ClimateControlPanel.init(containerElement, { tent_id: 1, ... });
 *   ClimateControlPanel.loadRecommendation(climateData);
 */
const ClimateControlPanel = (function() {
    let _container = null;
    let _tentId = null;
    let _currentClimate = null;
    let _recommendation = null;
    let _manualMode = false;
    let _options = {
        tent_id: null,
        show_action_buttons: true,
        show_projection: true,
        onActionApply: null,
        onTrain: null,
        api_base: '/api/v3/climate',
    };

    function init(container, options = {}) {
        _container = typeof container === 'string'
            ? document.querySelector(container)
            : container;

        _options = Object.assign({}, _options, options);
        _tentId = _options.tent_id;

        if (!_container) {
            console.error('ClimateControlPanel: container not found');
            return;
        }

        _render();
    }

    function _render() {
        _container.innerHTML = `
            <div class="climate-panel">
                <div class="panel-header">
                    <h3 class="section-title">
                        微气候调控
                        <span class="tent-label">帐篷 ${_tentId || '-'}</span>
                    </h3>
                    <div class="panel-tabs">
                        <button class="tab-btn active" data-tab="recommend" id="tabRecommend">推荐方案</button>
                        <button class="tab-btn" data-tab="manual" id="tabManual">手动调节</button>
                    </div>
                </div>

                <div class="panel-body">
                    <div class="climate-states">
                        <div class="state-card current">
                            <div class="state-label">当前状态</div>
                            <div class="state-metrics">
                                <div class="metric">
                                    <span class="metric-icon">🌡️</span>
                                    <span class="metric-value" id="curTemp">-</span>
                                    <span class="metric-unit">°C</span>
                                </div>
                                <div class="metric">
                                    <span class="metric-icon">💧</span>
                                    <span class="metric-value" id="curHum">-</span>
                                    <span class="metric-unit">%</span>
                                </div>
                                <div class="metric">
                                    <span class="metric-icon">☀️</span>
                                    <span class="metric-value" id="curLight">-</span>
                                    <span class="metric-unit">lux</span>
                                </div>
                                <div class="metric">
                                    <span class="metric-icon">⚗️</span>
                                    <span class="metric-value" id="curAw">-</span>
                                    <span class="metric-unit">Aw</span>
                                </div>
                            </div>
                        </div>

                        <div class="state-arrow" id="stateArrow">→</div>

                        <div class="state-card projected">
                            <div class="state-label">调控后投影</div>
                            <div class="state-metrics">
                                <div class="metric">
                                    <span class="metric-icon">🌡️</span>
                                    <span class="metric-value" id="projTemp">-</span>
                                    <span class="metric-unit">°C</span>
                                </div>
                                <div class="metric">
                                    <span class="metric-icon">💧</span>
                                    <span class="metric-value" id="projHum">-</span>
                                    <span class="metric-unit">%</span>
                                </div>
                                <div class="metric">
                                    <span class="metric-icon">☀️</span>
                                    <span class="metric-value" id="projLight">-</span>
                                    <span class="metric-unit">lux</span>
                                </div>
                                <div class="metric">
                                    <span class="metric-icon">⚗️</span>
                                    <span class="metric-value" id="projAw">-</span>
                                    <span class="metric-unit">Aw</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="action-section" id="actionSection">
                        <div class="action-label">推荐动作</div>
                        <div class="action-buttons" id="actionButtons">
                            <button class="action-btn vent-btn" data-action="vent">
                                <span class="action-icon">🌀</span>
                                <span class="action-text" id="ventLevel">通风: 关</span>
                            </button>
                            <button class="action-btn shade-btn" data-action="shade">
                                <span class="action-icon">⛱️</span>
                                <span class="action-text" id="shadeLevel">遮阳: 关</span>
                            </button>
                            <button class="action-btn humid-btn" data-action="humid">
                                <span class="action-icon">💨</span>
                                <span class="action-text" id="humidLevel">加湿: 关</span>
                            </button>
                        </div>
                        <div class="action-info">
                            <span class="energy-cost">能耗: <span id="energyCost">0</span> 单位</span>
                            <span class="reward">预期奖励: <span id="expectedReward">0</span></span>
                            <span class="shelf-life">保质期延长: <span id="shelfImprove">0</span> 天</span>
                        </div>
                        <div class="algorithm-tag" id="algoTag">算法: DQN</div>
                    </div>

                    <div class="control-actions">
                        <button class="btn btn-primary" id="btnApply">应用调控</button>
                        <button class="btn btn-secondary" id="btnTrain">训练模型</button>
                    </div>
                </div>
            </div>
        `;

        document.getElementById('tabRecommend').addEventListener('click', () => _switchTab('recommend'));
        document.getElementById('tabManual').addEventListener('click', () => _switchTab('manual'));
        document.getElementById('btnApply').addEventListener('click', _handleApply);
        document.getElementById('btnTrain').addEventListener('click', _handleTrain);

        document.querySelectorAll('.action-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                if (_manualMode) {
                    _cycleAction(btn.dataset.action);
                }
            });
        });
    }

    function _switchTab(tab) {
        _manualMode = tab === 'manual';

        document.querySelectorAll('.panel-tabs .tab-btn').forEach(b => {
            b.classList.toggle('active', b.dataset.tab === tab);
        });

        const label = document.querySelector('.action-label');
        if (label) label.textContent = _manualMode ? '手动调节' : '推荐动作';

        const btns = document.querySelectorAll('.action-btn');
        btns.forEach(b => b.style.cursor = _manualMode ? 'pointer' : 'default');
    }

    function _cycleAction(action) {
        if (!_recommendation) return;

        const actionData = { ..._recommendation.action };

        if (action === 'vent') {
            actionData.ventilation = (actionData.ventilation + 1) % 3;
        } else if (action === 'shade') {
            actionData.shading = (actionData.shading + 1) % 2;
        } else if (action === 'humid') {
            actionData.humidifier = (actionData.humidifier + 1) % 2;
        }

        _recommendation.action = actionData;
        _updateActionDisplay();
    }

    async function loadRecommendation(climate) {
        _currentClimate = climate;
        _updateCurrentState(climate);

        try {
            const resp = await fetch(`${_options.api_base}/recommend`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    tent_id: _tentId,
                    temperature: climate.temperature,
                    humidity: climate.humidity,
                    light: climate.light,
                    aw: climate.aw,
                }),
            });
            const data = await resp.json();
            setRecommendation(data);
        } catch (e) {
            console.error('ClimateControlPanel load failed:', e);
        }
    }

    function setRecommendation(data) {
        _recommendation = data;
        _updateProjectedState(data.projected_state);
        _updateActionDisplay();

        const algoTag = document.getElementById('algoTag');
        if (algoTag) {
            algoTag.textContent = `算法: ${data.algorithm === 'rule_based' ? '规则策略' : 'DQN'}`;
            algoTag.className = `algorithm-tag ${data.algorithm}`;
        }
    }

    function _updateCurrentState(climate) {
        const setVal = (id, val, decimals = 1) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val != null ? val.toFixed(decimals) : '-';
        };
        setVal('curTemp', climate?.temperature);
        setVal('curHum', climate?.humidity, 0);
        setVal('curLight', climate?.light, 0);
        setVal('curAw', climate?.aw, 2);
    }

    function _updateProjectedState(state) {
        const setVal = (id, val, decimals = 1) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val != null ? val.toFixed(decimals) : '-';
        };
        setVal('projTemp', state?.temperature);
        setVal('projHum', state?.humidity, 0);
        setVal('projLight', state?.light, 0);
        setVal('projAw', state?.aw, 2);
    }

    function _updateActionDisplay() {
        if (!_recommendation) return;
        const a = _recommendation.action;

        const ventTexts = ['关', '低', '高'];
        const shadeTexts = ['关', '开'];
        const humidTexts = ['关', '开'];

        document.getElementById('ventLevel').textContent = `通风: ${ventTexts[a.ventilation] || '关'}`;
        document.getElementById('shadeLevel').textContent = `遮阳: ${shadeTexts[a.shading] || '关'}`;
        document.getElementById('humidLevel').textContent = `加湿: ${humidTexts[a.humidifier] || '关'}`;

        document.getElementById('energyCost').textContent = (a.energy_cost || 0).toFixed(1);
        document.getElementById('expectedReward').textContent =
            (_recommendation.expected_reward || 0).toFixed(2);
        document.getElementById('shelfImprove').textContent =
            (_recommendation.shelf_life_improvement_days || 0).toFixed(1);

        document.querySelectorAll('.action-btn').forEach(btn => {
            const action = btn.dataset.action;
            const active = (action === 'vent' && a.ventilation > 0)
                || (action === 'shade' && a.shading > 0)
                || (action === 'humid' && a.humidifier > 0);
            btn.classList.toggle('active', active);
        });
    }

    function _handleApply() {
        if (_options.onActionApply && _recommendation) {
            _options.onActionApply(_recommendation);
        }
    }

    function _handleTrain() {
        if (_options.onTrain) {
            _options.onTrain();
        }
    }

    function getRecommendation() { return _recommendation; }
    function getCurrentClimate() { return _currentClimate; }

    return {
        init,
        loadRecommendation,
        setRecommendation,
        getRecommendation,
        getCurrentClimate,
    };
})();
