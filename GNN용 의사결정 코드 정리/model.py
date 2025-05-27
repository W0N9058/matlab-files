import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import MessagePassing


def generate_3d_grid(n_x, n_y, n_z):
    # 노드 좌표 생성, 3차원 격자 구조
    coords = [(x, y, z) for z in range(n_z) for y in range(n_y) for x in range(n_x)]
    idx = lambda x, y, z: x + y * n_x + z * n_x * n_y
    edges = []
    for x, y, z in coords:
        i = idx(x, y, z)
        for dx, dy, dz in [(1,0,0), (0,1,0), (0,0,1)]:
            nx_, ny_, nz_ = x+dx, y+dy, z+dz
            if 0 <= nx_ < n_x and 0 <= ny_ < n_y and 0 <= nz_ < n_z:
                j = idx(nx_, ny_, nz_)
                edges.append((i, j))
                edges.append((j, i))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


class ElectroThermalNodeModel(MessagePassing):
    # MessagePassing를 상속해서 메세지르를 처리하는 레이어 만듬
    # MessagePassing는 torch_geometric 안에 있음
    def __init__(self, in_elec=2, in_therm=5, in_env=2, hidden=32):
        super().__init__(aggr='add')
        
        ########### ψ 메시지 MLPs ###########
        # 전기 관련 메세지지
        self.psi_elec = torch.nn.Sequential(
            torch.nn.Linear(in_elec, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden)
        )
        # 내부 생성열 메세지지
        self.psi_gen = torch.nn.Sequential(
            torch.nn.Linear(in_therm, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden)
        )
        # 전도 열전달 메세지지
        self.psi_cond = torch.nn.Sequential(
            torch.nn.Linear(2, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden)
        )
        # 대류류 열전달 메세지
        self.psi_conv = torch.nn.Sequential(
            torch.nn.Linear(1, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden)
        )
        # 주변 환경 정보
        self.psi_env = torch.nn.Sequential(
            torch.nn.Linear(in_env, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden)
        )
        
        ########### φ 업데이트 MLPss ###########
        # 다음 SOC 산출
        self.phi_elec = torch.nn.Sequential(
            torch.nn.Linear(hidden + 1, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)
        )
        # 다음 온도 산출
        self.phi_therm = torch.nn.Sequential(
            torch.nn.Linear(hidden*4 + 1, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)
        )
        
        ########### ρ 리드아웃 MLPss ###########
        # SOC와 전류로 전압 예측
        self.rho_elec = torch.nn.Sequential(
            torch.nn.Linear(1 + 1, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)
        )
        # 온도로 최종 보정온도 산출
        self.rho_therm = torch.nn.Sequential(
            torch.nn.Linear(1, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)
        )

    def forward(self, x_elec, x_therm, x_env, edge_index, global_current=None):
        # x_elec: [N,1] SOC
        # x_therm: [N,5] [T,I,V,OCV,dUdT]
        # x_env: [N,2] [Surround_T, env_type]
        # global_current: [1,1]
        
        ####### 전기 메시지 #########
        # SOC랑 전류 묶어서 입력벡터 만들고, 첫번째 MLP인 psi_elec에 통과시키기 -> 전기 메세지 m_e 생성
        m_e = self.psi_elec(torch.cat([x_elec, global_current.expand_as(x_elec)], dim=-1))
        # x_elec + m_e로 입력 벡터 만들고, 다음 SOC인 soc_new 생성
        soc_new = self.phi_elec(torch.cat([x_elec, m_e], dim=-1))
        
        ####### 열 메시지  ######### 
        # 노드별 전기 열 정보를 담은 x_therm로 메세지(m_gen) 생성
        m_gen = self.psi_gen(x_therm)
        # 온도 따로 때어내기
        T = x_therm[:,0:1]
        # 전도 (이웃), x가 이웃 노드 온도
        m_cond = self.propagate(edge_index, x=(T, T))
        # 대류
        m_conv = self.psi_conv(T)
        # 전도나 대류 이외의 열교환 효과 고려
        m_env = self.psi_env(x_env)
        # 총 집계(단순히 더하는 식으로)
        Mt = m_gen + m_cond + m_conv + m_env
        # 온도 업데이트
        T_new = self.phi_therm(torch.cat([T, m_gen, m_cond, m_conv, m_env], dim=-1))
        
        #######  출력 예측  ######### 
        # SOC랑 전류 바탕으로 전압 예측
        V_pred = self.rho_elec(torch.cat([soc_new, global_current.expand_as(soc_new)], dim=-1))
        # 새로운 온도에 대한 보정
        T_pred = self.rho_therm(T_new)
        return soc_new, T_new, V_pred, T_pred

    # 주변 노드로 보내는 정보 계산
    def message(self, x_j, x_i):
        # 받는 노드, 보내는 노드 연결
        return self.psi_cond(torch.cat([x_i, x_j], dim=-1))


############ 임시 데이터 생성용 클래스 ############
class SyntheticBatteryDataset(Dataset):
    def __init__(self, num_graphs=100, grid_size=(10,10,5), seq_len=50):
        super().__init__()
        # 그래프 개수
        self.num_graphs = num_graphs
        # 격자 크기 (3차원)
        self.grid_size = grid_size
        # 시계열 길이
        self.seq_len = seq_len

    # 그래프 샘플 수 반환(체크용)
    def len(self):
        return self.num_graphs

    # 샘플 생성
    def get(self, idx):
        # 3D 격자 생성
        nx, ny, nz = self.grid_size
        edge_index = generate_3d_grid(nx, ny, nz)
        N = nx * ny * nz
        # 전기 상태
        SOC = torch.rand(N,1)
        I = torch.rand(N,1)*2-1
        V = torch.rand(N,1)
        # 열 상태
        OCV = torch.rand(N,1)
        dUdT = torch.rand(N,1)*1e-3
        T_init = torch.rand(N,1)*40 + 20
        x_therm = torch.cat([T_init, I, V, OCV, dUdT], dim=-1)
        # 환경 상태(수냉 파이프, 인접 배터리)
        Surround_T = torch.rand(N,1)*30 + 10
        env_type = torch.randint(0,3,(N,1)).float()  # 0: pipe, 1: adjacent battery, 2: none
        x_env = torch.cat([Surround_T, env_type], dim=-1)
        # 글로벌 전류
        global_I = torch.rand(1,1)*2 - 1
        # 데이터 객체 생성
        data = Data(
            x_elec=SOC,
            x_therm=x_therm,
            x_env=x_env,
            edge_index=edge_index,
            y_elec=V,
            y_therm=T_init,
            global_current=global_I
        )
        return data


def train():
    dataset = SyntheticBatteryDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    model = ElectroThermalNodeModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(20):
        loss_all = 0
        for batch in loader:
            soc, T, V_pred, T_pred = model(
                batch.x_elec, batch.x_therm, batch.x_env, batch.edge_index, global_current=batch.global_current
            )
            loss = F.mse_loss(V_pred, batch.y_elec) + F.mse_loss(T_pred, batch.y_therm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_all += loss.item()
        print(f"Epoch {epoch}, Loss: {loss_all:.4f}")

if __name__ == '__main__':
    train()
