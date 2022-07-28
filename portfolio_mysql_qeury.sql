CREATE VIEW view_port AS
select 
conv.gov_dn_cd,conv.signgu_nm,conv.adstrd_nm,conv.sex_se,conv.year_se, sum(conv.cnsmr_popltn_co)/92 as conv_popltn_co, #편의점 정보값 불러오기 
ifnull(i.ice_popltn_co,(select avg(ice.cnsmr_popltn_co) from ice where ice.bntr_nm like '서울%' and ice.year_se between 20 and 49 )) as ice_popltn_co , #아이스크림 값 가져오기 
ifnull(c.coffee_popltn_co,(select avg(coffee.cnsmr_popltn_co) from coffee where coffee.bntr_nm like '서울%' and coffee.year_se between 20 and 49 )) as coffee_popltn_co, # 커피값 가져오기
ifnull(income_r.income_avg,(select avg(income.ave_income_amt) from income where income.legaldong_nm like '송파%')) as income_avg_, #평균수입 가져오기 
ifnull(pop_f.pop_s_r,avg(pop_f.pop_s_r)) as pop_sum #인구값 가져오기 
from conv
#아이스크림 값 서브쿼리로 조인
left join 
	(select 
	ice.gov_dn_cd,	ice.signgu_nm,	ice.adstrd_nm,	ice.sex_se,	ice.year_se, sum(ice.cnsmr_popltn_co)/92 as ice_popltn_co
	from ice
	where
	ice.bntr_nm like '서울%'and
	ice.year_se between 20 and 49 
	group by ice.gov_dn_cd,ice.sex_se,ice.year_se) i
	on i.gov_dn_cd = conv.gov_dn_cd and i.sex_se = conv.sex_se and i.year_se=conv.year_se
#커피 값 서브쿼리로 조인 
left join 
	(select 
	coffee.gov_dn_cd,	coffee.signgu_nm,	coffee.adstrd_nm,	coffee.sex_se,	coffee.year_se, sum(coffee.cnsmr_popltn_co)/92 as coffee_popltn_co
	from coffee
	where
	coffee.bntr_nm like '서울%'and
	coffee.year_se between 20 and 49 
	group by coffee.gov_dn_cd,coffee.sex_se,coffee.year_se) c
	on c.gov_dn_cd = conv.gov_dn_cd and c.sex_se = conv.sex_se and c.year_se=conv.year_se
#평균 수입 서브쿼리 조인
left join
	(select income.adstrd_cd, avg(income.ave_income_amt) as income_avg from income group by income.adstrd_cd) income_r
    on income_r.adstrd_cd = conv.gov_dn_cd
#인구수 서브쿼리 조인
left join
	#mysql에는 python 처럼 melt 함수가 없기 때문에 union을 이용했다.
	(select pop_s.legaldong_nm,(select case when pop_s.sex_se = '남자' then 'M' else 'F' end) as sex_se_p,20 as 'year_se',pop_s.y20_24 as pop_s_r   from pop_s where pop_s.adstrd_nm not like '%계' and pop_s.legaldong_nm not like '%계' and pop_s.sex_se not like '%계'
	union
	select pop_s.legaldong_nm,(select case when pop_s.sex_se = '남자' then 'M' else 'F' end) as sex_se_p,25 as 'year_se',pop_s.y25_29 as pop_s_r   from pop_s where pop_s.adstrd_nm not like '%계' and pop_s.legaldong_nm not like '%계' and pop_s.sex_se not like '%계'
	union
	select pop_s.legaldong_nm,(select case when pop_s.sex_se = '남자' then 'M' else 'F' end) as sex_se_p,30 as 'year_se',pop_s.y30_34 as pop_s_r   from pop_s where pop_s.adstrd_nm not like '%계' and pop_s.legaldong_nm not like '%계' and pop_s.sex_se not like '%계'
	union
	select pop_s.legaldong_nm,(select case when pop_s.sex_se = '남자' then 'M' else 'F' end) as sex_se_p,35 as 'year_se',pop_s.y35_39 as pop_s_r   from pop_s where pop_s.adstrd_nm not like '%계' and pop_s.legaldong_nm not like '%계' and pop_s.sex_se not like '%계'
	union
	select pop_s.legaldong_nm,(select case when pop_s.sex_se = '남자' then 'M' else 'F' end) as sex_se_p,40 as 'year_se',pop_s.y40_44 as pop_s_r   from pop_s where pop_s.adstrd_nm not like '%계' and pop_s.legaldong_nm not like '%계' and pop_s.sex_se not like '%계'
	union
	select pop_s.legaldong_nm,(select case when pop_s.sex_se = '남자' then 'M' else 'F' end) as sex_se_p,45 as 'year_se',pop_s.y45_49 as pop_s_r   from pop_s where pop_s.adstrd_nm not like '%계' and pop_s.legaldong_nm not like '%계' and pop_s.sex_se not like '%계')  pop_f
    on pop_f.legaldong_nm = conv.adstrd_nm and pop_f.sex_se_p = conv.sex_se and pop_f.year_se = conv.year_se
where
conv.bntr_nm like '서울%'and
conv.year_se between 20 and 49
group by conv.gov_dn_cd,conv.sex_se,conv.year_se
;
