/**
 * RouteSlider - 商队路径选择滑块组件
 * 支持路线选择、季节切换，实时展示药品损耗对比
 * 带有GIS热力图联动和推荐评分展示
 *
 * 用法：
 *   RouteSlider.init(containerElement, { routes: [...], ... });
 *   RouteSlider.setOnChange(handler);
 */
const RouteSlider = (function() {
    let _container = null;
    let _routes = [];
    let _seasons = [];
    let _selectedRoute = null;
    let _selectedSeason = null;
    let _analysisData = null;
    let _options = {
        routes: [],
        seasons: [],
        show_season_selector: true,
        show_loss_comparison: true,
        show_heatmap_preview: true,
        onRouteChange: null,
        onSeasonChange: null,
        api_base: '/api/v3/route',
    };

    function init(container, options = {}) {
        _container = typeof container === 'string'
            ? document.querySelector(container)
            : container;

        _options = Object.assign({}, _options, options);
        _routes = _options.routes || [];
        _seasons = _options.seasons || [];

        if (_routes.length > 0) {
            _selectedRoute = _routes[0].name || _routes[0];
        }

        if (!_container) {
            console.error('RouteSlider: container not found');
            return;
        }

        _render();
        if (_routes.length === 0 || _seasons.length === 0) {
            _loadMetadata();
        }
    }

    function _render() {
        _container.innerHTML = `
            <div class="route-slider-panel">
                <div class="panel-header">
                    <h3 class="section-title">商队路径分析</h3>
                    ${_options.show_season_selector ? `
                        <div class="season-selector">
                            <label>季节:</label>
                            <select id="routeSeason" class="form-select">
                                ${_seasons.map(s => `<option value="${s}">${s}</option>`).join('')}
                            </select>
                        </div>
                    ` : ''}
                </div>

                <div class="route-slider-container">
                    <div class="route-slider-track" id="routeTrack">
                        ${_routes.map((r, i) => `
                            <div class="route-slider-marker ${i === 0 ? 'active' : ''}"
                                 data-route="${r.name || r}"
                                 data-index="${i}">
                                <div class="marker-dot"></div>
                                <div class="marker-label">${r.name || r}</div>
                            </div>
                        `).join('')}
                        <div class="slider-connect" id="sliderConnect"></div>
                    </div>
                    <input type="range" id="routeSliderInput"
                           min="0" max="${Math.max(0, _routes.length - 1)}"
                           value="0" class="route-slider-input">
                </div>

                <div class="route-details" id="routeDetails">
                    <div class="detail-grid">
                        <div class="detail-item">
                            <span class="detail-label">总距离</span>
                            <span class="detail-value" id="detailDistance">-</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">行程天数</span>
                            <span class="detail-value" id="detailDays">-</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">平均温度</span>
                            <span class="detail-value" id="detailTemp">-</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">推荐评分</span>
                            <span class="detail-value highlight" id="detailScore">-</span>
                        </div>
                    </div>
                </div>

                ${_options.show_loss_comparison ? `
                    <div class="loss-comparison">
                        <div class="comparison-header">药品损耗预估</div>
                        <div class="comparison-bars" id="lossBars">
                            <div class="empty-state">加载中...</div>
                        </div>
                    </div>
                ` : ''}

                <div class="route-footer">
                    <button class="btn btn-primary" id="btnAnalyze">详细分析</button>
                    <span class="route-status" id="routeStatus">准备就绪</span>
                </div>
            </div>
        `;

        const slider = document.getElementById('routeSliderInput');
        if (slider) {
            slider.addEventListener('input', _handleSliderChange);
            slider.addEventListener('change', _handleSliderChangeEnd);
        }

        document.querySelectorAll('.route-slider-marker').forEach(marker => {
            marker.addEventListener('click', () => {
                const idx = parseInt(marker.dataset.index);
                _setRouteIndex(idx);
            });
        });

        const seasonSel = document.getElementById('routeSeason');
        if (seasonSel) {
            seasonSel.addEventListener('change', _handleSeasonChange);
        }

        const btnAnalyze = document.getElementById('btnAnalyze');
        if (btnAnalyze) {
            btnAnalyze.addEventListener('click', _handleAnalyze);
        }
    }

    async function _loadMetadata() {
        try {
            const [routesResp, seasonsResp] = await Promise.all([
                fetch(`${_options.api_base}/routes`),
                fetch(`${_options.api_base}/seasons`),
            ]);

            if (routesResp.ok) {
                const data = await routesResp.json();
                _routes = data.routes || [];
                if (_routes.length > 0 && !_selectedRoute) {
                    _selectedRoute = _routes[0].name;
                }
            }

            if (seasonsResp.ok) {
                const data = await seasonsResp.json();
                _seasons = data.seasons || [];
                if (_seasons.length > 0 && !_selectedSeason) {
                    _selectedSeason = _seasons[0];
                }
            }

            _updateSlider();
            if (_selectedRoute) {
                _loadAnalysis();
            }
        } catch (e) {
            console.error('RouteSlider metadata load failed:', e);
        }
    }

    function _handleSliderChange(e) {
        const idx = parseInt(e.target.value);
        const route = _routes[idx];
        if (route) {
            _updateMarkerActive(idx);
            _selectedRoute = route.name || route;
        }
    }

    function _handleSliderChangeEnd(e) {
        _handleSliderChange(e);
        _loadAnalysis();

        if (_options.onRouteChange) {
            _options.onRouteChange(_selectedRoute);
        }
    }

    function _handleSeasonChange(e) {
        _selectedSeason = e.target.value;
        _loadAnalysis();

        if (_options.onSeasonChange) {
            _options.onSeasonChange(_selectedSeason);
        }
    }

    function _setRouteIndex(idx) {
        const slider = document.getElementById('routeSliderInput');
        if (slider) slider.value = idx;

        const route = _routes[idx];
        if (route) {
            _selectedRoute = route.name || route;
            _updateMarkerActive(idx);
            _loadAnalysis();

            if (_options.onRouteChange) {
                _options.onRouteChange(_selectedRoute);
            }
        }
    }

    function _updateSlider() {
        const slider = document.getElementById('routeSliderInput');
        if (slider) {
            slider.max = Math.max(0, _routes.length - 1);
        }

        const track = document.getElementById('routeTrack');
        if (track) {
            track.innerHTML = _routes.map((r, i) => `
                <div class="route-slider-marker ${i === 0 ? 'active' : ''}"
                     data-route="${r.name || r}"
                     data-index="${i}" style="left: ${_getMarkerPosition(i)}%;">
                    <div class="marker-dot"></div>
                    <div class="marker-label">${r.name || r}</div>
                </div>
            `).join('') + '<div class="slider-connect" id="sliderConnect"></div>';

            track.querySelectorAll('.route-slider-marker').forEach(marker => {
                marker.addEventListener('click', () => {
                    const idx = parseInt(marker.dataset.index);
                    _setRouteIndex(idx);
                });
            });
        }
    }

    function _getMarkerPosition(idx) {
        if (_routes.length <= 1) return 50;
        return (idx / (_routes.length - 1)) * 100;
    }

    function _updateMarkerActive(idx) {
        document.querySelectorAll('.route-slider-marker').forEach((m, i) => {
            m.classList.toggle('active', i === idx);
        });

        const connect = document.getElementById('sliderConnect');
        if (connect) {
            const pct = _getMarkerPosition(idx);
            connect.style.width = `${pct}%`;
        }
    }

    async function _loadAnalysis() {
        if (!_selectedRoute) return;

        const statusEl = document.getElementById('routeStatus');
        if (statusEl) statusEl.textContent = '加载中...';

        try {
            const season = encodeURIComponent(_selectedSeason || '');
            const resp = await fetch(
                `${_options.api_base}/analyze/${encodeURIComponent(_selectedRoute)}?season=${season}`
            );
            const data = await resp.json();
            _analysisData = data;
            _updateDetails(data);
            _updateLossBars(data.drug_losses || []);

            if (statusEl) statusEl.textContent = '分析完成';
        } catch (e) {
            console.error('Route analysis load failed:', e);
            if (statusEl) statusEl.textContent = '加载失败';
        }
    }

    function _updateDetails(data) {
        const setText = (id, val, suffix = '') => {
            const el = document.getElementById(id);
            if (el) el.textContent = val != null ? val + suffix : '-';
        };

        setText('detailDistance', data.total_distance_km?.toFixed(0), ' km');
        setText('detailDays', data.total_travel_days?.toFixed(0), ' 天');
        setText('detailTemp', data.exposure?.avg_temperature?.toFixed(1), '°C');
        setText('detailScore', data.recommendation_score?.toFixed(0));

        const scoreEl = document.getElementById('detailScore');
        if (scoreEl && data.recommendation_score != null) {
            scoreEl.className = 'detail-value ' +
                (data.recommendation_score >= 70 ? 'positive' :
                 data.recommendation_score >= 40 ? 'neutral' : 'negative');
        }
    }

    function _updateLossBars(drugLosses) {
        const container = document.getElementById('lossBars');
        if (!container) return;

        if (!drugLosses || drugLosses.length === 0) {
            container.innerHTML = '<div class="empty-state">暂无数据</div>';
            return;
        }

        const maxLoss = Math.max(...drugLosses.map(d => d.total_loss_pct || 0), 1);

        container.innerHTML = drugLosses.map(d => `
            <div class="loss-bar-item">
                <div class="loss-bar-label">${d.drug_name}</div>
                <div class="loss-bar-track">
                    <div class="loss-bar-fill ${d.total_loss_pct > 30 ? 'high' : d.total_loss_pct > 15 ? 'mid' : 'low'}"
                         style="width: ${(d.total_loss_pct / maxLoss) * 100}%;"></div>
                </div>
                <div class="loss-bar-value">${(d.total_loss_pct || 0).toFixed(1)}%</div>
            </div>
        `).join('');
    }

    function _handleAnalyze() {
        if (_analysisData && _options.onRouteChange) {
            _options.onRouteChange(_selectedRoute, _analysisData);
        }
    }

    function getSelectedRoute() { return _selectedRoute; }
    function getSelectedSeason() { return _selectedSeason; }
    function getAnalysisData() { return _analysisData; }

    function setRoute(routeName) {
        const idx = _routes.findIndex(r => (r.name || r) === routeName);
        if (idx >= 0) _setRouteIndex(idx);
    }

    function setSeason(season) {
        _selectedSeason = season;
        const sel = document.getElementById('routeSeason');
        if (sel) sel.value = season;
        _loadAnalysis();
    }

    return {
        init,
        getSelectedRoute,
        getSelectedSeason,
        getAnalysisData,
        setRoute,
        setSeason,
    };
})();
