import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { GraphComponent } from './graph.component';
import { TuiMapperPipeModule } from '@taiga-ui/cdk';
import { TuiDataListModule, TuiDropdownModule } from '@taiga-ui/core';

@NgModule({
  imports: [
    CommonModule,
    TuiMapperPipeModule,
    TuiDropdownModule,
    TuiDataListModule
  ],
  declarations: [GraphComponent],
  exports: [GraphComponent]
})
export class GraphModule { }
