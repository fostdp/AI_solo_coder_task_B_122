/**
 * AllocationTable - 药材调配优化表格组件
 * 展示各帐篷间的药品调配方案，支持按药品/帐篷筛选
 *
 * 用法：
 *   AllocationTable.init(containerElement, options);
 *   AllocationTable.loadData(apiUrl);
 */
const AllocationTable = (function() {
    let _container = null;
    let _data = null;
    let _options = {
        onAllocationClick: null,
        showTentSummaries: true,
        maxRows: 20,
    };

    function init(container, options = {}) {
        _container = typeof container === 'string'
            ? document.querySelector(container)
            : container;

        _options = Object.assign({}, _options, options);

        if (!_container) {
            console.error('AllocationTable: container not found');
            return;
        }

        _render();
    }

    function _render() {
        _container.innerHTML = `
            <div class="allocation-table-wrapper">
                <div class="allocation-header">
                    <h3 class="section-title">药材调配方案</h3>
                    <div class="allocation-filters">
                        <select id="allocFilterDrug" class="form-select">
                            <option value="">全部药品</option>
                        </select>
                        <select id="allocFilterTent" class="form-select">
                            <option value="">全部帐篷</option>
                        </select>
                    </div>
                </div>
                <div class="allocation-summary" id="allocSummary" style="display:none;">
                    <div class="summary-item">
                        <span class="summary-label">调配方案数</span>
                        <span class="summary-value" id="allocCount">0</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">预计减少浪费</span>
                        <span class="summary-value positive" id="allocWasteRed">0</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">运输成本</span>
                        <span class="summary-value" id="allocTransCost">0</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-value method-tag" id="allocMethod">-</span>
                    </div>
                </div>
                <div class="table-container">
                    <table class="allocation-table" id="allocTable">
                        <thead>
                            <tr>
                                <th>药品</th>
                                <th>调出帐篷</th>
                                <th>调入帐篷</th>
                                <th>数量</th>
                                <th>减少浪费</th>
                                <th>原因</th>
                            </tr>
                        </thead>
                        <tbody id="allocTableBody">
                            <tr><td colspan="6" class="empty-state">加载中...</td></tr>
                        </tbody>
                    </table>
                </div>
                <div class="table-footer" id="allocFooter" style="display:none;">
                    <span class="text-muted">显示前 <span id="allocShowing">0</span> 条</span>
                </div>
            </div>
        `;

        document.getElementById('allocFilterDrug').addEventListener('change', _applyFilters);
        document.getElementById('allocFilterTent').addEventListener('change', _applyFilters);
    }

    async function loadData(apiUrl) {
        const tbody = document.getElementById('allocTableBody');
        if (tbody) tbody.innerHTML = '<tr><td colspan="6" class="empty-state">加载中...</td></tr>';

        try {
            const resp = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ drug_risks: _generateSampleRisks() }),
            });
            const data = await resp.json();
            setData(data);
        } catch (e) {
            console.error('AllocationTable load failed:', e);
            if (tbody) {
                tbody.innerHTML = '<tr><td colspan="6" class="empty-state">加载失败</td></tr>';
            }
        }
    }

    function _generateSampleRisks() {
        const drugs = ['当归', '大黄', '甘草', '黄芪', '白术'];
        const risks = [];
        for (let i = 1; i <= 5; i++) {
            drugs.forEach((drug, idx) => {
                const shelf = i <= 2 ? 10 + idx * 3 : 100 + idx * 20;
                risks.push({
                    tent_id: i,
                    drug_name: drug,
                    shelf_life_days: shelf,
                    current_stock: 200,
                    daily_consumption: 2.0,
                });
            });
        }
        return risks;
    }

    function setData(data) {
        _data = data;
        _updateSummary();
        _populateFilters();
        _applyFilters();
    }

    function _updateSummary() {
        const summary = document.getElementById('allocSummary');
        if (!summary || !_data) return;

        summary.style.display = 'flex';
        document.getElementById('allocCount').textContent = _data.allocations?.length || 0;
        document.getElementById('allocWasteRed').textContent =
            (_data.total_waste_reduction || 0).toFixed(1) + ' 单位';
        document.getElementById('allocTransCost').textContent =
            (_data.total_transport_cost || 0).toFixed(1);

        const methodEl = document.getElementById('allocMethod');
        if (_data.method) {
            methodEl.textContent = _data.method === 'column_generation' ? '列生成法' : '全量MIP';
            methodEl.title = `求解方法: ${_data.method}`;
        } else if (_data.status) {
            methodEl.textContent = _data.status;
        }
    }

    function _populateFilters() {
        if (!_data?.allocations) return;

        const drugFilter = document.getElementById('allocFilterDrug');
        const tentFilter = document.getElementById('allocFilterTent');
        const drugs = new Set();
        const tents = new Set();

        _data.allocations.forEach(a => {
            drugs.add(a.drug_name);
            tents.add(a.from_tent);
            tents.add(a.to_tent);
        });

        drugs.forEach(d => {
            const opt = document.createElement('option');
            opt.value = d; opt.textContent = d;
            drugFilter.appendChild(opt);
        });

        Array.from(tents).sort((a, b) => a - b).forEach(t => {
            const opt = document.createElement('option');
            opt.value = t; opt.textContent = `帐篷 ${t}`;
            tentFilter.appendChild(opt);
        });
    }

    function _applyFilters() {
        const tbody = document.getElementById('allocTableBody');
        const footer = document.getElementById('allocFooter');
        if (!tbody || !_data?.allocations) return;

        const drugFilter = document.getElementById('allocFilterDrug').value;
        const tentFilter = document.getElementById('allocFilterTent').value;

        let filtered = _data.allocations.filter(a => {
            if (drugFilter && a.drug_name !== drugFilter) return false;
            if (tentFilter &&
                String(a.from_tent) !== tentFilter &&
                String(a.to_tent) !== tentFilter) return false;
            return true;
        });

        const showCount = Math.min(filtered.length, _options.maxRows);
        const shown = filtered.slice(0, showCount);

        if (shown.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="empty-state">暂无调配方案</td></tr>';
            footer.style.display = 'none';
            return;
        }

        tbody.innerHTML = shown.map(a => `
            <tr class="alloc-row" data-drug="${a.drug_name}">
                <td><strong>${a.drug_name}</strong></td>
                <td><span class="tent-badge tent-out">${a.from_tent}号</span></td>
                <td><span class="tent-badge tent-in">${a.to_tent}号</span></td>
                <td>${a.quantity.toFixed(1)} 单位</td>
                <td class="positive">-${(a.estimated_waste_reduction || 0).toFixed(1)}</td>
                <td class="text-muted small">${a.reason || '-'}</td>
            </tr>
        `).join('');

        footer.style.display = 'block';
        document.getElementById('allocShowing').textContent =
            `${showCount} / ${filtered.length}`;

        if (_options.onAllocationClick) {
            tbody.querySelectorAll('.alloc-row').forEach(row => {
                row.addEventListener('click', () => {
                    const idx = Array.from(tbody.children).indexOf(row);
                    _options.onAllocationClick(shown[idx]);
                });
            });
        }
    }

    function getData() { return _data; }

    return {
        init,
        loadData,
        setData,
        getData,
    };
})();
