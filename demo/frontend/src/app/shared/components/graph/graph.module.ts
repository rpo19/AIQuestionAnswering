import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { GraphComponent } from './graph.component';
import { TuiMapperPipeModule } from '@taiga-ui/cdk';

@NgModule({
  imports: [
    CommonModule,
    TuiMapperPipeModule
  ],
  declarations: [GraphComponent],
  exports: [GraphComponent]
})
export class GraphModule { }
