/**
 * SubstituteBadge - 方剂替代推荐徽章组件
 * 展示某味药材的替代推荐，支持展开/收起详情，毒性警示
 *
 * 用法：
 *   SubstituteBadge.init(containerElement, { herb_name: '当归', ... });
 *   SubstituteBadge.loadRecommendations(apiUrl);
 */
const SubstituteBadge = (function() {
    let _container = null;
    let _herbName = '';
    let _recommendations = [];
    let _expanded = false;
    let _options = {
        herb_name: '',
        max_display: 3,
        show_toxicity_warning: true,
        onSubstituteSelect: null,
        api_base: '/api/v3/herb',
    };

    function init(container, options = {}) {
        _container = typeof container === 'string'
            ? document.querySelector(container)
            : container;

        _options = Object.assign({}, _options, options);
        _herbName = _options.herb_name || '';

        if (!_container) {
            console.error('SubstituteBadge: container not found');
            return;
        }

        _render();
    }

    function _render() {
        _container.innerHTML = `
            <div class="substitute-badge">
                <div class="substitute-header" id="subHeader">
                    <span class="herb-name">${_herbName || '药材'}</span>
                    <span class="substitute-count" id="subCount">0 种替代</span>
                    <span class="expand-icon" id="subExpand">▼</span>
                </div>
                <div class="substitute-content" id="subContent" style="display:none;">
                    <div class="substitute-list" id="subList">
                        <div class="empty-state">加载中...</div>
                    </div>
                    <div class="substitute-footer" id="subFooter" style="display:none;">
                        <span class="text-muted small">
                            基于《千金方》知识图谱推荐
                        </span>
                    </div>
                </div>
            </div>
        `;

        document.getElementById('subHeader').addEventListener('click', toggle);
    }

    async function loadRecommendations(apiUrl) {
        const listEl = document.getElementById('subList');
        if (listEl) listEl.innerHTML = '<div class="empty-state">加载中...</div>';

        try {
            const url = apiUrl || `${_options.api_base}/substitutes/${encodeURIComponent(_herbName)}`;
            const resp = await fetch(url);
            const data = await resp.json();
            setRecommendations(data.recommendations || []);
        } catch (e) {
            console.error('SubstituteBadge load failed:', e);
            if (listEl) {
                listEl.innerHTML = '<div class="empty-state">加载失败</div>';
            }
        }
    }

    function setRecommendations(recs) {
        _recommendations = recs;
        document.getElementById('subCount').textContent = `${recs.length} 种替代`;
        _renderList();
    }

    function _renderList() {
        const listEl = document.getElementById('subList');
        const footerEl = document.getElementById('subFooter');
        if (!listEl) return;

        if (_recommendations.length === 0) {
            listEl.innerHTML = '<div class="empty-state">暂无替代药材</div>';
            footerEl.style.display = 'none';
            return;
        }

        const displayCount = Math.min(_recommendations.length, _options.max_display);
        const items = _recommendations.slice(0, displayCount);

        listEl.innerHTML = items.map(rec => {
            const toxicity = rec.toxicity || '无毒';
            const toxicClass = _getToxicityClass(toxicity);
            const toxicBadge = _options.show_toxicity_warning && toxicity !== '无毒'
                ? `<span class="toxic-badge ${toxicClass}">${toxicity}</span>` : '';

            const scorePct = Math.round((rec.similarity_score || 0) * 100);
            const pathLabel = _getPathLabel(rec.path_type);

            return `
                <div class="substitute-item" data-herb="${rec.substitute_herb}">
                    <div class="sub-item-header">
                        <span class="sub-herb-name">${rec.substitute_herb}</span>
                        ${toxicBadge}
                        <span class="sub-score">匹配度 ${scorePct}%</span>
                    </div>
                    <div class="sub-item-details">
                        <span class="sub-path">${pathLabel}</span>
                        ${rec.shared_efficacy?.length
                            ? `<span class="sub-efficacy">共效: ${rec.shared_efficacy.slice(0, 2).join('、')}</span>`
                            : ''}
                    </div>
                    ${rec.notes ? `<div class="sub-item-notes">${rec.notes}</div>` : ''}
                    ${rec.available_in_tents?.length
                        ? `<div class="sub-item-avail">
                            储备帐篷: ${rec.available_in_tents.map(t => `${t}号`).join('、')}
                           </div>` : ''}
                </div>
            `;
        }).join('');

        footerEl.style.display = 'block';

        if (_options.onSubstituteSelect) {
            listEl.querySelectorAll('.substitute-item').forEach(item => {
                item.addEventListener('click', () => {
                    const herb = item.dataset.herb;
                    const rec = _recommendations.find(r => r.substitute_herb === herb);
                    _options.onSubstituteSelect(rec);
                });
            });
        }
    }

    function _getToxicityClass(toxicity) {
        const classes = {
            '无毒': 'toxic-none',
            '小毒': 'toxic-low',
            '有毒': 'toxic-mid',
            '大毒': 'toxic-high',
        };
        return classes[toxicity] || 'toxic-unknown';
    }

    function _getPathLabel(pathType) {
        const labels = {
            '同类': '同分类',
            '互补': '功效互补',
            '配伍': '经典配伍',
            '替代': '临床替代',
        };
        return labels[pathType] || pathType || '相关';
    }

    function toggle() {
        _expanded = !_expanded;
        const content = document.getElementById('subContent');
        const icon = document.getElementById('subExpand');
        if (content) content.style.display = _expanded ? 'block' : 'none';
        if (icon) icon.textContent = _expanded ? '▲' : '▼';
    }

    function expand() { if (!_expanded) toggle(); }
    function collapse() { if (_expanded) toggle(); }

    function getRecommendations() { return _recommendations; }
    function getHerbName() { return _herbName; }

    return {
        init,
        loadRecommendations,
        setRecommendations,
        toggle,
        expand,
        collapse,
        getRecommendations,
        getHerbName,
    };
})();
