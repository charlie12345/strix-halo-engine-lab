#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strix {
inline bool cohort_default_small_group(std::uint32_t rows, bool enabled) { return enabled && rows <= 8U; }
// Shared by the default route and every budget group of at most eight rows.
// Fits preserves the live verifier admission check, including padding repair.
template<class Item, class Decline, class Fits, class Execute>
void cohort_default_small_plan(std::vector<Item> items, std::size_t member_count,
    Fits fits, Execute execute) {
        std::vector<Item> solo;
        std::vector<Decline> why(member_count,Decline::none); // a solo session's reason
        std::vector<std::vector<Item>> groups;
        std::stable_sort(items.begin(),items.end(),[](const Item& a,const Item& b){return a.rows>b.rows;});
        for(const auto& item:items) {
            if(!item.rows || item.rows>=8U){
                solo.push_back(item);why[item.member]=item.rows?Decline::wide:Decline::no_rows;continue;
            }
            bool placed=false;
            for(auto& g:groups) {
                std::uint32_t sum=0;for(const auto& x:g)sum+=x.rows;
                if(sum+item.rows<=8U){g.push_back(item);placed=true;break;}
            }
            if(!placed)groups.push_back({item});
        }
        for(auto& g:groups) {
            for(;;) {
                if(g.size()<2U) break;
                std::uint32_t sum=0;for(const auto& x:g)sum+=x.rows;
                if(sum==4U || sum==8U) break;
                // Round up to the next cohort size with the padding rows of a smaller window.
                const auto target=sum<4U?4U:8U;
                bool grown=false;
                for(auto x=g.rbegin();x!=g.rend() && !grown;++x)
                    for(const auto rows:{x->rows*2U,x->rows*4U})
                        if(rows<=8U && sum-x->rows+rows==target && fits(*x,rows)){x->rows=rows;grown=true;break;}
                if(!grown){why[g.back().member]=Decline::padding;solo.push_back(g.back());g.pop_back();}
            }
            if(g.size()==1U){why[g.front().member]=Decline::alone;solo.push_back(g.front());g.clear();}
        }
    execute(groups, solo, why);
}
} // namespace strix
