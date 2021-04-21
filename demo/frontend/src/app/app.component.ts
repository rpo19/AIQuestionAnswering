import { trigger, transition, style, animate } from '@angular/animations';
import { Component, HostBinding, OnInit } from '@angular/core';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss'],
  animations: [
    trigger('enterAnimation', [
      transition(':enter', [
        style({ transform: 'translateY(10%)', opacity: 0}),
        animate('500ms ease-out', style({ transform: 'translateY(0)', opacity: 1}))
      ])
    ])
  ]
})
export class AppComponent implements OnInit {

  constructor() {}

  ngOnInit() { }
  
}
